from typing import Dict

from ConfigSpace.configuration_space import ConfigurationSpace

from dswizard.components.base import ComponentChoice, EstimatorComponent
from dswizard.components.pipeline import FlexiblePipeline
from dswizard.core.base_structure_generator import BaseStructureGenerator
from dswizard.core.model import CandidateStructure


class FixedStructure(BaseStructureGenerator):

    def __init__(self, steps: Dict[str, EstimatorComponent], dataset_properties: dict, timeout: int = None):
        super().__init__(dataset_properties=dataset_properties, timeout=timeout)
        self.configspace = ConfigurationSpace()

        for step, task in steps.items():
            if isinstance(task, ComponentChoice) or isinstance(task, EstimatorComponent):
                cs = task.get_hyperparameter_search_space(dataset_properties=dataset_properties)
            else:
                raise ValueError('Unable to handle type {}'.format(type(task)))
            self.configspace.add_configuration_space(step, cs)

        self.pipeline = FlexiblePipeline(steps, dataset_properties)

    def get_candidate(self, budget: float = None) -> CandidateStructure:
        return CandidateStructure(self.configspace, self.pipeline, budget, timeout=self.timeout, model_based_pick=False)
