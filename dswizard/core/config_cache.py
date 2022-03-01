from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Dict, List
from typing import Type, Tuple

import joblib
import numpy as np
from ConfigSpace import ConfigurationSpace
from ConfigSpace.configuration_space import Configuration
from sklearn.pipeline import Pipeline

from dswizard.core.model import ConfigKey, CandidateId
from dswizard.core.similaritystore import SimilarityStore
from dswizard.util import autoproxy

if TYPE_CHECKING:
    from dswizard.core.base_config_generator import BaseConfigGenerator
    from dswizard.core.model import Job
    from dswizard.components.meta_features import MetaFeatures

autoproxy.apply()


class ConfigCache:
    class Entry:

        def __init__(self, model: Pipeline):
            self.store = SimilarityStore(model)
            self.generators: List[BaseConfigGenerator] = []

        def add(self, mf, cg):
            self.store.add(mf)
            self.generators.append(cg)
            return len(self.generators) - 1

    def __init__(self,
                 clazz: Type[BaseConfigGenerator],
                 model: str = None,
                 init_kwargs: Dict = None,
                 logger: logging.Logger = None):
        self.clazz = clazz

        try:
            if model is not None:
                with open(model, 'rb') as f:
                    self.model, _ = joblib.load(f)
            else:
                self.model = None
        except FileNotFoundError:
            self.model = None

        self.init_kwargs = init_kwargs
        self.cache: Dict[float, ConfigCache.Entry] = {}

        if logger is None:
            self.logger = logging.getLogger('ConfigCache')
        else:
            self.logger = logger

    def get_config_key(self,
                       configspace: ConfigurationSpace = None,
                       mf: MetaFeatures = None,
                       max_distance: float = 0.05,
                       **kwargs) -> ConfigKey:
        if configspace is None or mf is None:
            raise ValueError('If cfg_key is not given, both configspace and mf must not be None.')

        hash_key = hash(configspace)
        if hash_key not in self.cache:
            self.cache[hash_key] = ConfigCache.Entry(self.model)
            cg = self.clazz(configspace, **{**self.init_kwargs, **kwargs})
            idx = self.cache[hash_key].add(mf, cg)
            return ConfigKey(hash_key, idx)

        entry = self.cache[hash_key]
        distance, idx, _ = entry.store.get_similar(mf)
        if distance <= max_distance:
            return ConfigKey(hash_key, int(idx))
        else:
            cg = self.clazz(configspace, **{**self.init_kwargs, **kwargs})
            idx = self.cache[hash_key].add(mf, cg)
            return ConfigKey(hash_key, idx)

    def sample_configuration(self,
                             cid: CandidateId = None,
                             cfg_key: ConfigKey = None,
                             name: str = None,
                             configspace: ConfigurationSpace = None,
                             mf: np.ndarray = None,
                             max_distance: float = 1,
                             default: bool = False, **kwargs) -> Tuple[Configuration, ConfigKey]:
        if cfg_key is None:
            cfg_key = self.get_config_key(configspace, mf, max_distance, **kwargs)
        cg = self.cache[cfg_key.hash].generators[cfg_key.idx]
        config = cg.sample_config(cid=cid, cfg_key=cfg_key, name=name, default=default)
        return config, cfg_key

    def explain(self):
        res = {}
        for hash_, entry in self.cache.items():
            for idx, gen in enumerate(entry.generators):
                res[ConfigKey(hash_, idx)] = gen.explain()
        return res

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
        except Exception:
            self.logger.exception("Failed to register results")
