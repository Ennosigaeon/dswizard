from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, Type, List, Tuple
from typing import TYPE_CHECKING

import Pyro4
from ConfigSpace import ConfigurationSpace
from ConfigSpace.configuration_space import Configuration

from dswizard.core.model import MetaFeatures
from dswizard.util.singleton import Singleton

if TYPE_CHECKING:
    from dswizard.core.base_config_generator import BaseConfigGenerator
    from dswizard.core.model import Job, PartialConfig, Result


class PyroDaemon:

    def __init__(self,
                 nameserver: str = None,
                 nameserver_port: int = None,
                 host: str = None,
                 run_id: str = '0',
                 logger: logging.Logger = None
                 ):
        self.nameserver = nameserver
        self.nameserver_port = nameserver_port
        self.host = host
        self.run_id = run_id
        self.pyro_daemon = None
        self.pyro_id = None

        if logger is None:
            self.logger = logging.getLogger('ConfigCache')
        else:
            self.logger = logger

    def run(self):
        if self.nameserver is None:
            return

        self.pyro_daemon = Pyro4.core.Daemon(host=self.host)

        with Pyro4.locateNS(host=self.nameserver, port=self.nameserver_port) as ns:
            uri = self.pyro_daemon.register(self, self.pyro_id)
            ns.register(self.pyro_id, uri)

        self.logger.info('Pyro daemon running on {}'.format(self.pyro_daemon.locationStr))

        self.pyro_daemon.requestLoop()

        with Pyro4.locateNS(self.nameserver, port=self.nameserver_port) as ns:
            ns.remove(self.pyro_id)

    def shutdown(self) -> None:
        if self.pyro_daemon is not None:
            self.pyro_daemon.shutdown()


@Singleton
class ConfigCache(PyroDaemon):

    def __init__(self,
                 clazz: Type[BaseConfigGenerator],
                 init_kwargs: dict,
                 nameserver: str = None,
                 nameserver_port: int = None,
                 host: str = None,
                 run_id: str = '0',
                 logger: logging.Logger = None):
        super().__init__(nameserver, nameserver_port, host, run_id, logger)

        self.clazz = clazz
        self.init_kwargs = init_kwargs
        self.cache: Dict[ConfigurationSpace, BaseConfigGenerator] = {}

        self.result_cache: Dict[float, Dict[str, List[Tuple[PartialConfig, Result]]]] = defaultdict(
            lambda: defaultdict(list))

    @Pyro4.expose
    def register_result(self, job: Job):
        try:
            self.cache[job.pipeline.configuration_space].register_result(job)
        except KeyError:
            pass

        budget = job.budget
        configs = job.result.partial_configs
        result = job.result

        for config in configs:
            self.result_cache[budget][config.estimator].append((config, result))

    @Pyro4.expose
    def get_trainings_data(self, estimator: str, meta: MetaFeatures, budget: float) -> \
            List[Tuple[Configuration, float]]:
        if budget is None and len(self.result_cache.keys()) > 0:
            budget = max(self.result_cache.keys())

        results = self.result_cache[budget][estimator]

        ls = []
        for partial_config, res in results:
            if meta.similar(partial_config.meta):
                ls.append((partial_config.configuration, res.loss))

        self.logger.debug('Found {} similar samples'.format(len(ls)))
        return ls

    @Pyro4.expose
    def get_config_generator(self, configspace: ConfigurationSpace = None, **kwargs) -> BaseConfigGenerator:
        if configspace not in self.cache:
            cg = self.clazz(configspace, **{**self.init_kwargs, **kwargs})
            self.cache[configspace] = cg
        return self.cache[configspace]
