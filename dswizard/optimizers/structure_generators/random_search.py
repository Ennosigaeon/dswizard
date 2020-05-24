import math
import random
from typing import Tuple, List

import numpy as np
from ConfigSpace import ConfigurationSpace

from automl.components import classification, data_preprocessing, feature_preprocessing
from automl.components.base import EstimatorComponent
from automl.components.classification import ClassifierChoice
from automl.components.data_preprocessing import DataPreprocessorChoice
from automl.components.feature_preprocessing import FeaturePreprocessorChoice
from dswizard.components.pipeline import FlexiblePipeline, SubPipeline
from dswizard.core.base_structure_generator import BaseStructureGenerator
from dswizard.core.model import CandidateStructure, Dataset


class RandomStructureGenerator(BaseStructureGenerator):

    # noinspection PyProtectedMember
    def __init__(self,
                 max_depth: int = 10,
                 include_basic_estimators: bool = False,
                 **kwargs):
        super().__init__(**kwargs)

        self.max_depth = max_depth

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

    def _determine_depth(self, n_min: int = 1, n_max: int = 2):
        r = int(math.ceil(np.random.normal(0.5, 0.5 / 3) * n_max))
        return max(min(self.max_depth, r), n_min)

    def get_candidate(self, ds: Dataset) -> CandidateStructure:
        attempts = 1
        while True:
            try:
                depth = self._determine_depth(n_max=self.max_depth)
                cs, steps = self._generate_pipeline(depth)

                pipeline = FlexiblePipeline(steps)

                print(steps)
                self.logger.debug('Created valid pipeline after {} tries'.format(attempts))
                return CandidateStructure(cs, pipeline, model_based_pick=False)
            except TypeError:
                attempts += 1

    def _generate_pipeline(self, depth: int) -> \
            Tuple[ConfigurationSpace, List[Tuple[str, EstimatorComponent]]]:
        cs = ConfigurationSpace()
        steps = []
        i = 0

        while i < depth:
            name = 'step_{}'.format(i)
            clazz = random.sample(self.candidates, 1)[0]

            if clazz == SubPipeline.name():
                max_depth = depth - i - 1
                instance, n = self._generate_subpipelines(max_depth)

                # Do not add SubPipeline without any sub-steps
                if n == 0:
                    i += 1
                    continue
                else:
                    i += n
            else:
                instance = self._get_estimator_instance(clazz)

            steps.append((name, instance))
            cs.add_configuration_space(name, instance.get_hyperparameter_search_space())

            i += 1
        return cs, steps

    def _generate_subpipelines(self, max_depth: int) -> Tuple[SubPipeline, int]:
        n_pipelines = random.choice([2, 3, 4])

        depths = []
        pipelines = []
        for i in range(n_pipelines):
            d = self._determine_depth(0, max_depth)
            depths.append(d)

            cs, steps = self._generate_pipeline(d)
            pipelines.append(steps)
        pipelines = list(filter(lambda d: len(d) > 0, pipelines))
        return SubPipeline(pipelines), max(depths)
