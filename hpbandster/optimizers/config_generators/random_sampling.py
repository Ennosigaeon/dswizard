from typing import Tuple

from ConfigSpace.configuration_space import Configuration

from hpbandster.core import BaseConfigGenerator, ConfigInfo


class RandomSampling(BaseConfigGenerator):
    """
    class to implement random sampling from a ConfigSpace
    """

    def __init__(self, **kwargs):
        """
        :param kwargs: see hyperband.core.BaseConfigGenerator for additional arguments
        """

        super().__init__(**kwargs)

    def get_config(self, budget: float) -> Tuple[Configuration, ConfigInfo]:
        if self.configspace is None:
            raise ValueError('No configuration space provided. Call set_config_space(ConfigurationSpace) first.')

        return self.configspace.sample_configuration(), ConfigInfo(
            model_based_pick=False
        )
