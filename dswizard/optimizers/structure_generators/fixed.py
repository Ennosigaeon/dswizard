from typing import List, Tuple

import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace

from automl.components.base import ComponentChoice, EstimatorComponent
from dswizard.components.pipeline import FlexiblePipeline
from dswizard.core.base_structure_generator import BaseStructureGenerator
from dswizard.core.config_cache import ConfigCache
from dswizard.core.model import CandidateStructure, Dataset


class FixedStructure(BaseStructureGenerator):

    def __init__(self, steps: List[Tuple[str, EstimatorComponent]], cfg_cache: ConfigCache, **kwargs):
        super().__init__(cfg_cache=cfg_cache, **kwargs)
        self.configspace = ConfigurationSpace()
        self.cfg_keys: List[Tuple[float, int]] = []

        length = 0
        for step, task in steps:
            if isinstance(task, ComponentChoice) or isinstance(task, EstimatorComponent):
                cg, key = cfg_cache.get_config_generator(configspace=task.get_hyperparameter_search_space(),
                                                         mf=np.ones((1, 1)) * length)
                cs = task.get_hyperparameter_search_space()
            else:
                raise ValueError('Unable to handle type {}'.format(type(task)))
            self.configspace.add_configuration_space(step, cs)
            self.cfg_keys.append(key)
            length += 1

        self.pipeline = FlexiblePipeline(steps)

    def get_candidate(self, ds: Dataset) -> CandidateStructure:
        return CandidateStructure(self.configspace, self.pipeline, self.cfg_keys, model_based_pick=False)
