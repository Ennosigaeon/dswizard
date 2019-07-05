from collections import OrderedDict
from typing import Tuple

from ConfigSpace.configuration_space import Configuration, ConfigurationSpace

from core import BaseConfigGenerator
from hpbandster.core import BaseStructureGenerator


class DummyStructure(BaseStructureGenerator):

    def __init__(self, config_generator: BaseConfigGenerator):
        super().__init__(config_generator)

    def get_config(self, budget: float) -> Tuple[Configuration, dict]:
        config, info = self.config_generator.get_config(budget)
        structure = OrderedDict()
        structure['dummy'] = ''
        info.structure = structure
        return config, info

    def get_config_space(self) -> ConfigurationSpace:
        return self.config_generator.configspace
