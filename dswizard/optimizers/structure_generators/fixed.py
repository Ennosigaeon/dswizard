from typing import List, Tuple

from dswizard.components.base import ComponentChoice, EstimatorComponent
from dswizard.core.base_structure_generator import BaseStructureGenerator
from dswizard.core.config_cache import ConfigCache
from dswizard.core.model import CandidateStructure, Dataset
from dswizard.pipeline.pipeline import FlexiblePipeline


class FixedStructure(BaseStructureGenerator):

    def __init__(self, steps: List[Tuple[str, EstimatorComponent]], cfg_cache: ConfigCache, **kwargs):
        super().__init__(cfg_cache=cfg_cache, **kwargs)
        self.steps = steps

    def fill_candidate(self, cs: CandidateStructure, ds: Dataset, **kwargs) -> CandidateStructure:
        cfg_keys: List[Tuple[float, int]] = []
        for step, task in self.steps:
            if isinstance(task, ComponentChoice) or isinstance(task, EstimatorComponent):
                cg, key = self.cfg_cache.get_config_generator(configspace=task.get_hyperparameter_search_space(),
                                                              mf=ds.meta_features)
            else:
                raise ValueError('Unable to handle type {}'.format(type(task)))
            cfg_keys.append(key)

        cs.pipeline = FlexiblePipeline(self.steps)
        cs.configspace = cs.pipeline.get_hyperparameter_search_space(ds.meta_features)
        cs.cfg_keys = cfg_keys
        return cs
