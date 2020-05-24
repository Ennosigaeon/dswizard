from typing import List, Tuple

from ConfigSpace.configuration_space import ConfigurationSpace

from automl.components.base import ComponentChoice, EstimatorComponent
from dswizard.components.pipeline import FlexiblePipeline
from dswizard.core.base_structure_generator import BaseStructureGenerator
from dswizard.core.model import CandidateStructure
from dswizard.core.model import Dataset


class FixedStructure(BaseStructureGenerator):

    def __init__(self, steps: List[Tuple[str, EstimatorComponent]]):
        super().__init__()
        self.configspace = ConfigurationSpace()

        for step, task in steps:
            if isinstance(task, ComponentChoice) or isinstance(task, EstimatorComponent):
                cs = task.get_hyperparameter_search_space()
            else:
                raise ValueError('Unable to handle type {}'.format(type(task)))
            self.configspace.add_configuration_space(step, cs)

        self.pipeline = FlexiblePipeline(steps)

    def get_candidate(self, ds: Dataset) -> CandidateStructure:
        return CandidateStructure(self.configspace, self.pipeline, model_based_pick=False)
