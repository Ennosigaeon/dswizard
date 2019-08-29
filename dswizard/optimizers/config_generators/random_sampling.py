from ConfigSpace.configuration_space import Configuration

from dswizard.core.base_config_generator import BaseConfigGenerator


class RandomSampling(BaseConfigGenerator):
    """
    class to implement random sampling from a ConfigSpace
    """

    def get_config(self, budget: float = None) -> Configuration:
        if self.configspace is None:
            raise ValueError('No configuration space provided. Call set_config_space(ConfigurationSpace) first.')

        return self.configspace.sample_configuration()

    def get_config_for_step(self, step: str, budget: float = None) -> Configuration:
        raise NotImplementedError('RandomSampling does not support JIT configuration')
