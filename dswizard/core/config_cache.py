from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, Type
from typing import TYPE_CHECKING

from ConfigSpace import ConfigurationSpace
from ConfigSpace.configuration_space import Configuration

from dswizard.core.model import MetaFeatures

if TYPE_CHECKING:
    from dswizard.core.base_config_generator import BaseConfigGenerator
    from dswizard.core.model import Job


# @Singleton
class ConfigCache:

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

        # TODO add budget again
        # self.cache: Dict[float, Dict[ConfigurationSpace, Dict[MetaFeatures, BaseConfigGenerator]]] = defaultdict(
        #     lambda: defaultdict(lambda: defaultdict()))
        self.cache: Dict[ConfigurationSpace, Dict[MetaFeatures, BaseConfigGenerator]] = defaultdict(
            lambda: defaultdict())

    def get_config_generator(self, budget: float, configspace: ConfigurationSpace, meta_features: MetaFeatures,
                             **kwargs) -> BaseConfigGenerator:
        try:
            for mf, cg in self.cache[configspace].items():
                if mf.similar(meta_features):
                    return self.cache[configspace][meta_features]
            else:
                cg = self.clazz(configspace, **{**self.init_kwargs, **kwargs})
                self.cache[configspace][meta_features] = cg
                return cg
        except Exception as ex:
            self.logger.exception(ex)
            raise ex

    def sample_configuration(self, budget: float, configspace: ConfigurationSpace, meta_features: MetaFeatures,
                             **kwargs) -> Configuration:
        return self.get_config_generator(budget, configspace, meta_features, **kwargs).sample_config()

    # noinspection PyUnresolvedReferences
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
                            config.configuration, loss, status)
            else:
                self.cache[job.pipeline.configuration_space][mf].register_result(job.config, loss, status)
        except Exception as ex:
            self.logger.exception(ex)
            raise ex
