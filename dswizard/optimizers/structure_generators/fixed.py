from typing import Tuple

from ConfigSpace.configuration_space import ConfigurationSpace

from dswizard.components.base import ComponentChoice, EstimatorComponent
from dswizard.core.base_structure_generator import BaseStructureGenerator
from dswizard.core.model import Structure


class FixedStructure(BaseStructureGenerator):

    def __init__(self, dataset_properties: dict, structure: Structure):
        super().__init__()
        self.configspace = ConfigurationSpace()

        for step, task in structure.items():
            if isinstance(task, ComponentChoice):
                cs = task.get_hyperparameter_search_space(dataset_properties=dataset_properties)
            elif isinstance(task, EstimatorComponent):
                cs = task.get_hyperparameter_search_space(dataset_properties=dataset_properties)
            else:
                raise ValueError('Unable to handle type {}'.format(type(task)))
            self.configspace.add_configuration_space(step, cs)
        self.structure = structure

    def get_config_space(self) -> Tuple[ConfigurationSpace, Structure]:
        return self.configspace, self.structure
