from typing import Tuple

from ConfigSpace.configuration_space import ConfigurationSpace

from hpbandster.core.base_structure_generator import BaseStructureGenerator
from hpbandster.core.model import Structure


class DummyStructure(BaseStructureGenerator):

    def __init__(self, configspace: ConfigurationSpace):
        super().__init__()
        self.configspace = configspace

    def get_config_space(self) -> Tuple[ConfigurationSpace, Structure]:
        return self.config_generator.configspace, {}
