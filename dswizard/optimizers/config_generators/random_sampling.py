from typing import Tuple

import numpy as np
from ConfigSpace.configuration_space import Configuration, ConfigurationSpace

from dswizard.components.pipeline import SubPipeline
from dswizard.core.base_config_generator import BaseConfigGenerator
from dswizard.core.model import MetaFeatures


class RandomSampling(BaseConfigGenerator):
    """
    class to implement random sampling from a ConfigSpace
    """

    def get_config(self, budget: float = None) -> Configuration:
        if self.configspace is None:
            raise ValueError('No configuration space provided. Call set_config_space(ConfigurationSpace) first.')

        return self.configspace.sample_configuration()

    def get_config_for_step(self, estimator: str, cs: ConfigurationSpace, X: np.ndarray, budget: float = None) -> \
            Tuple[Configuration, MetaFeatures]:
        meta_features = MetaFeatures(X)

        if estimator == SubPipeline.name():
            return Configuration(ConfigurationSpace(), {}), meta_features

        return cs.sample_configuration(), meta_features
