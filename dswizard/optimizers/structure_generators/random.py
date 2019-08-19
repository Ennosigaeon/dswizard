import random
from collections import OrderedDict
from typing import Tuple, Dict

import math
import numpy as np
from ConfigSpace import ConfigurationSpace

from dswizard.components import classification, data_preprocessing, feature_preprocessing
from dswizard.components.base import EstimatorComponent
from dswizard.components.classification import ClassifierChoice
from dswizard.components.data_preprocessing import DataPreprocessorChoice
from dswizard.components.feature_preprocessing import FeaturePreprocessorChoice
from dswizard.components.pipeline import FlexiblePipeline, SubPipeline
from dswizard.core.base_structure_generator import BaseStructureGenerator
from dswizard.core.model import CandidateStructure


class RandomStructureGenerator(BaseStructureGenerator):

    # noinspection PyProtectedMember
    def __init__(self,
                 dataset_properties: dict,
                 timeout: int = None,
                 max_steps: int = 10,
                 include_basic_estimators: bool = False):
        super().__init__(dataset_properties=dataset_properties, timeout=timeout)

        self.max_steps = max_steps

        self.candidates = {
            ClassifierChoice.name(),
            DataPreprocessorChoice.name(),
            FeaturePreprocessorChoice.name(),
            SubPipeline.name()
        }

        if include_basic_estimators:
            for estimator in classification._classifiers.values():
                self.candidates.add(estimator.name())
            for estimator in data_preprocessing._preprocessors.values():
                self.candidates.add(estimator.name())
            for estimator in feature_preprocessing._preprocessors.values():
                self.candidates.add(estimator.name())

    def _determine_n_steps(self, n_min: int = 1, n_max: int = 2):
        r = int(math.ceil(np.random.normal(0.5, 0.5 / 3) * n_max))
        return max(min(self.max_steps, r), n_min)

    def get_candidate(self, budget: float) -> CandidateStructure:
        attempts = 1
        while True:
            try:
                n_steps = self._determine_n_steps(n_max=self.max_steps)
                cs, steps = self._generate_pipeline(n_steps)

                pipeline = FlexiblePipeline(steps, self.dataset_properties)

                self.logger.debug('Created valid pipeline after {} tries'.format(attempts))
                return CandidateStructure(cs, pipeline, budget, timeout=self.timeout, model_based_pick=False)
            except TypeError:
                attempts += 1

    def _generate_pipeline(self, n_steps: int) -> \
            Tuple[ConfigurationSpace, Dict[str, EstimatorComponent]]:
        cs = ConfigurationSpace()
        steps = OrderedDict()
        i = 0

        while i < n_steps:
            name = 'step_{}'.format(i)
            clazz = random.sample(self.candidates, 1)[0]

            if clazz == SubPipeline.name():
                max_steps = n_steps - i - 1
                instance, n = self._generate_subpipelines(max_steps)

                # Do not add SubPipeline without any sub-steps
                if n == 0:
                    i += 1
                    continue
                else:
                    i += n
            else:
                instance = self._get_estimator_instance(clazz)

            steps[name] = instance
            cs.add_configuration_space(name, instance.get_hyperparameter_search_space(self.dataset_properties))

            i += 1
        return cs, steps

    def _generate_subpipelines(self, max_steps: int) -> Tuple[SubPipeline, int]:
        n_pipelines = random.choice([2, 3, 4])

        actual_steps = 0
        pipelines = []
        for i in range(n_pipelines):
            n_steps = self._determine_n_steps(0, max_steps - actual_steps)
            actual_steps += n_steps

            cs, steps = self._generate_pipeline(n_steps)
            pipelines.append(steps)
        pipelines = list(filter(lambda d: len(d) > 0, pipelines))
        return SubPipeline(pipelines, self.dataset_properties), actual_steps
