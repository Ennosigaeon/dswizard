from typing import Tuple

from ConfigSpace.configuration_space import Configuration

from dswizard.core.base_config_generator import BaseConfigGenerator
from dswizard.core.model import ConfigInfo


class RandomSampling(BaseConfigGenerator):
    """
    class to implement random sampling from a ConfigSpace
    """

    def get_config(self, budget: float) -> Tuple[Configuration, ConfigInfo]:
        if self.configspace is None:
            raise ValueError('No configuration space provided. Call set_config_space(ConfigurationSpace) first.')

        return self.configspace.sample_configuration(), ConfigInfo(
            model_based_pick=False,
            structure=self.structure
        )
