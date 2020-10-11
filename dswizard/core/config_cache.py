from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict
from typing import Type, Tuple

import numpy as np
from ConfigSpace import ConfigurationSpace
from ConfigSpace.configuration_space import Configuration
from sklearn.neighbors import NearestNeighbors

if TYPE_CHECKING:
    from dswizard.core.base_config_generator import BaseConfigGenerator
    from dswizard.core.model import Job
    from automl.components.meta_features import MetaFeatures


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
                 logger: logging.Logger = None):

        self.clazz = clazz
        self.init_kwargs = init_kwargs

        if logger is None:
            self.logger = logging.getLogger('ConfigCache')
        else:
            self.logger = logger

        self.cache: Dict[float, ConfigCache.Entry] = {}

    def get_config_generator(self,
                             cfg_key: Tuple[float, int] = None,
                             configspace: ConfigurationSpace = None,
                             mf: MetaFeatures = None,
                             max_distance: float = 1, **kwargs) -> Tuple[BaseConfigGenerator, Tuple[float, int]]:
        if cfg_key is not None:
            return self.cache[cfg_key[0]].generators[cfg_key[1]], cfg_key

        if configspace is None or mf is None:
            raise ValueError('If cfg_key is not given, both configspace and mf must not be None.')

        hash_key = hash(configspace)
        if hash_key not in self.cache:
            self.cache[hash_key] = ConfigCache.Entry()
            cg = self.clazz(configspace, **{**self.init_kwargs, **kwargs})
            cg, idx = self.cache[hash_key].add(mf, cg)
            return cg, (hash_key, idx)

        entry = self.cache[hash_key]
        distance, idx = entry.neighbours.kneighbors(mf, n_neighbors=1)
        if distance[0][0] <= max_distance:
            return entry.generators[idx[0][0]], (hash_key, int(idx[0][0]))
        else:
            cg = self.clazz(configspace, **{**self.init_kwargs, **kwargs})
            cg, idx = self.cache[hash_key].add(mf, cg)
            return cg, (hash_key, idx)

    def sample_configuration(self,
                             cfg_key: Tuple[float, int] = None,
                             configspace: ConfigurationSpace = None,
                             mf: np.ndarray = None,
                             max_distance: float = 1, default: bool = False, **kwargs) \
            -> Tuple[Configuration, Tuple[float, int]]:
        cg, key = self.get_config_generator(cfg_key, configspace, mf, max_distance, **kwargs)
        return cg.sample_config(default=default), key

    # noinspection PyUnresolvedReferences
    def register_result(self, job: Job) -> None:
        try:
            loss = job.result.loss
            status = job.result.status

            if loss is None:
                return

            if len(job.result.partial_configs) > 0:
                for config in job.result.partial_configs:
                    if config.cfg_key is None or config.is_empty():
                        continue
                    self.cache[config.cfg_key[0]].generators[config.cfg_key[1]] \
                        .register_result(config.config, loss, status)
            else:
                cfg_key = job.cfg_keys[0]
                self.cache[cfg_key[0]].generators[cfg_key[1]].register_result(job.config, loss, status)
        except Exception as ex:
            self.logger.exception(ex)
            raise ex
