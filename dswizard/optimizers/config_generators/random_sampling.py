from ConfigSpace.configuration_space import Configuration

from dswizard.core.base_config_generator import BaseConfigGenerator


class RandomSampling(BaseConfigGenerator):
    """
    class to implement random sampling from a ConfigSpace
    """

    def sample_config(self, default: bool = False, **kwargs) -> Configuration:
        if self.configspace is None:
            raise ValueError('No configuration space provided. Call set_config_space(ConfigurationSpace) first.')

        if default:
            config = self.configspace.get_default_configuration()
            config.origin = 'Default'
            return config
        config = self.configspace.sample_configuration()
        config.origin = 'Random Search'
        return config
