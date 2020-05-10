from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict
from typing import Type, List, Tuple

import numpy as np
from ConfigSpace import ConfigurationSpace
from ConfigSpace.configuration_space import Configuration
from sklearn.neighbors import NearestNeighbors

if TYPE_CHECKING:
    from dswizard.core.base_config_generator import BaseConfigGenerator
    from dswizard.core.model import Job


class ConfigCache:
    class Entry:

        def __init__(self):
            self.mfs = None
            self.generators = []
            self.neighbours = NearestNeighbors()

        def add(self, mf, cg):
            if self.mfs is None:
                self.mfs = mf
            else:
                self.mfs = np.append(self.mfs, mf, axis=0)
            self.neighbours.fit(self.mfs)
            self.generators.append(cg)
            return cg, len(self.generators) - 1

    def __init__(self,
                 clazz: Type[BaseConfigGenerator],
                 init_kwargs: dict,
                 run_id: str = '0',
                 logger: logging.Logger = None):

        self.run_id = run_id
        self.clazz = clazz
        self.init_kwargs = init_kwargs

        if logger is None:
            self.logger = logging.getLogger('ConfigCache')
        else:
            self.logger = logger

        self.cache: Dict[float, ConfigCache.Entry] = {}

        self.mfs: Dict[float, np.ndarray] = {}
        self.generators: Dict[float, List[BaseConfigGenerator]] = {}
        self.neighbours = NearestNeighbors()

    def get_config_generator(self, budget: float, configspace: ConfigurationSpace, meta_features: np.ndarray,
                             max_distance: float = 1, **kwargs) -> Tuple[BaseConfigGenerator, int]:
        if budget not in self.cache:
            self.cache[budget] = ConfigCache.Entry()
            cg = self.clazz(configspace, **{**self.init_kwargs, **kwargs})
            return self.cache[budget].add(meta_features, cg)

        entry = self.cache[budget]
        distance, idx = entry.neighbours.kneighbors(meta_features, n_neighbors=1)
        if distance[0][0] <= max_distance:
            return entry.generators[idx[0][0]], int(idx[0][0])
        else:
            cg = self.clazz(configspace, **{**self.init_kwargs, **kwargs})
            return entry.add(meta_features, cg)

    def sample_configuration(self, budget: float, configspace: ConfigurationSpace, meta_features: np.ndarray,
                             max_distance: float = 1, **kwargs) -> Tuple[Configuration, int]:
        cg, idx = self.get_config_generator(budget, configspace, meta_features, max_distance, **kwargs)
        return cg.sample_config(), idx

    # noinspection PyUnresolvedReferences
    def register_result(self, job: Job) -> None:
        try:
            entry = self.cache[job.budget]
            loss = job.result.loss
            status = job.result.status

            if len(job.result.partial_configs) > 0:
                for config in job.result.partial_configs:
                    if not config.is_empty():
                        entry.generators[config.cfg_idx].register_result(config.config, loss, status)
            else:
                entry.generators[job.cfg_idx].register_result(job.config, loss, status)
        except Exception as ex:
            self.logger.exception(ex)
            raise ex
