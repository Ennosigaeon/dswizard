from typing import Tuple, Dict

from ConfigSpace import ConfigurationSpace
from ConfigSpace.configuration_space import Configuration

from hpbandster.core import BaseConfigGenerator, ConfigInfo


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

        super().__init__(configspace, **kwargs)

    def get_config(self, budget: float) -> Tuple[Configuration, ConfigInfo]:
        return self.configspace.sample_configuration(), ConfigInfo(
            model_based_pick=False
        )
