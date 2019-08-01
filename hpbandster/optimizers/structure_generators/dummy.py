from collections import OrderedDict
from typing import Tuple

from ConfigSpace.configuration_space import Configuration, ConfigurationSpace

from core import BaseConfigGenerator, ConfigInfo
from hpbandster.core import BaseStructureGenerator


class DummyStructure(BaseStructureGenerator):

    def __init__(self, configspace: ConfigurationSpace):
        super().__init__()
        self.configspace = configspace

    def set_config_generator(self, config_generator: BaseConfigGenerator):
        super().set_config_generator(config_generator)
        config_generator.set_config_space(self.configspace)

    def get_config(self, budget: float) -> Tuple[Configuration, ConfigInfo]:
        config, info = self.config_generator.get_config(budget)
        structure = OrderedDict()
        structure['dummy'] = ''
        info.structure = structure
        return config, info

    def get_config_space(self) -> ConfigurationSpace:
        return self.config_generator.configspace
