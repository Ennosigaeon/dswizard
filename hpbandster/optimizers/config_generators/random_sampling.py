from typing import Tuple

from ConfigSpace import ConfigurationSpace

from hpbandster.core.base_config_generator import BaseConfigGenerator


class RandomSampling(BaseConfigGenerator):
    """
        class to implement random sampling from a ConfigSpace
    """

    def __init__(self, configspace: ConfigurationSpace, **kwargs):
        """

        Parameters:
        -----------

        configspace: ConfigSpace.ConfigurationSpace
            The configuration space to sample from. It contains the full
            specification of the Hyperparameters with their priors
        **kwargs:
            see  hyperband.config_generators.base.base_config_generator for additional arguments
        """

        super().__init__(**kwargs)
        self.configspace = configspace

    def get_config(self, budget: float) -> Tuple[dict, dict]:
        return self.configspace.sample_configuration().get_dictionary(), {}
