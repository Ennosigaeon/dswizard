from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, Type
from typing import TYPE_CHECKING

import Pyro4
from ConfigSpace import ConfigurationSpace
from ConfigSpace.configuration_space import Configuration

from dswizard.core.model import MetaFeatures

if TYPE_CHECKING:
    from dswizard.core.base_config_generator import BaseConfigGenerator
    from dswizard.core.model import Job


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


# @Singleton
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

        # TODO add budget again
        # self.cache: Dict[float, Dict[ConfigurationSpace, Dict[MetaFeatures, BaseConfigGenerator]]] = defaultdict(
        #     lambda: defaultdict(lambda: defaultdict()))
        self.cache: Dict[ConfigurationSpace, Dict[MetaFeatures, BaseConfigGenerator]] = defaultdict(lambda: defaultdict())

    @Pyro4.expose
    def get_config_generator(self, budget: float, configspace: ConfigurationSpace, meta_features: MetaFeatures,
                             **kwargs) -> BaseConfigGenerator:
        try:
            # TODO BaseConfigGenerator objects are not shared across processes
            for mf, cg in self.cache[configspace].items():
                if mf.similar(meta_features):
                    return self.cache[configspace][meta_features]
            else:
                # cg = self.mgr.ConfigGenerator(configspace, **{**self.init_kwargs, **kwargs})
                cg = self.clazz(configspace, **{**self.init_kwargs, **kwargs})
                self.cache[configspace][meta_features] = cg
                return cg
        except Exception as ex:
            self.logger.exception(ex)
            raise ex

    @Pyro4.expose
    def sample_configuration(self, budget: float, configspace: ConfigurationSpace, meta_features: MetaFeatures,
                             **kwargs) -> Configuration:
        return self.get_config_generator(budget, configspace, meta_features, **kwargs).sample_config(budget)

    # noinspection PyUnresolvedReferences
    @Pyro4.expose
    def register_result(self, job: Job) -> None:
        try:
            budget = job.budget
            mf = job.ds.meta_features
            loss = job.result.loss
            status = job.result.status

            if len(job.result.partial_configs) > 0:
                for config in job.result.partial_configs:
                    if len(config.configuration.configuration_space.get_hyperparameters()) > 0:
                        self.cache[config.configuration.configuration_space][mf].register_result(
                            config.configuration, loss, status, budget)
            else:
                self.cache[job.pipeline.configuration_space][mf].register_result(job.config, loss,
                                                                                         status, budget)
        except Exception as ex:
            self.logger.exception(ex)
            raise ex
