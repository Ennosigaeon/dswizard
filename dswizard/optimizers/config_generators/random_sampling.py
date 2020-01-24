from ConfigSpace.configuration_space import Configuration

from dswizard.core.base_config_generator import BaseConfigGenerator


class RandomSampling(BaseConfigGenerator):
    """
    class to implement random sampling from a ConfigSpace
    """

    def sample_config(self, budget: float = None) -> Configuration:
        if self.configspace is None:
            raise ValueError('No configuration space provided. Call set_config_space(ConfigurationSpace) first.')

        return self.configspace.sample_configuration()
