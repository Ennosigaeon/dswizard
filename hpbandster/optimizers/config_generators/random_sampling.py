from typing import Tuple

from ConfigSpace import ConfigurationSpace

from hpbandster.core import BaseConfigGenerator


class RandomSampling(BaseConfigGenerator):
    """
    class to implement random sampling from a ConfigSpace
    """

    def __init__(self, configspace: ConfigurationSpace, **kwargs):
        """

        :param configspace: The configuration space to sample from. It contains the full specification of the
            Hyperparameters with their priors
        :param kwargs: see hyperband.core.BaseConfigGenerator for additional arguments
        """

        super().__init__(**kwargs)
        self.configspace = configspace

    def get_config(self, budget: float) -> Tuple[dict, dict]:
        return self.configspace.sample_configuration(), {}
