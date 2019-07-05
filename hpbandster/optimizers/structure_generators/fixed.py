from typing import Tuple, Dict, Union

from ConfigSpace.configuration_space import Configuration, ConfigurationSpace

from hpbandster.components.base import ComponentChoice, EstimatorComponent
from hpbandster.core import BaseStructureGenerator, ConfigInfo


class FixedStructure(BaseStructureGenerator):

    def __init__(self, dataset_properties: dict, structure: Dict[str, Union[ComponentChoice, EstimatorComponent]]):
        """

        :param task: Constant defined in hpbandster.components.constants
        :param config_generator:
        """

        # TODO build configuration space based on pipeline structure
        configspace = ConfigurationSpace()

        for step, task in structure.items():
            if isinstance(task, ComponentChoice):
                cs = task.get_hyperparameter_search_space(dataset_properties=dataset_properties)
            elif isinstance(task, EstimatorComponent):
                cs = task.get_hyperparameter_search_space(dataset_properties=dataset_properties)
            else:
                raise ValueError('Unable to handle type {}'.format(type(task)))
            configspace.add_configuration_space(step, cs)
        self.structure = structure

        super().__init__(configspace)

    def get_config(self, budget: float) -> Tuple[Configuration, ConfigInfo]:
        config, info = self.config_generator.get_config(budget)

        info.structure = self.structure
        return config, info
