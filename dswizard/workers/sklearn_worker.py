import importlib
import warnings
from typing import Optional, Tuple, Union, List

import numpy as np
from ConfigSpace import Configuration
from sklearn import clone
from sklearn.base import is_classifier

from automl.components.base import EstimatorComponent
from dswizard.components.pipeline import FlexiblePipeline
from dswizard.core.config_cache import ConfigCache
from dswizard.core.logger import ProcessLogger
from dswizard.core.model import CandidateId, Dataset
from dswizard.core.worker import Worker
from dswizard.util import util

warnings.filterwarnings("ignore", category=UserWarning)


class SklearnWorker(Worker):

    def compute(self,
                ds: Dataset,
                config_id: CandidateId,
                config: Optional[Configuration],
                cfg_cache: Optional[ConfigCache],
                cfg_keys: Optional[List[Tuple[float, int]]],
                pipeline: FlexiblePipeline,
                process_logger: ProcessLogger) -> List[float]:
        cloned_pipeline = clone(pipeline)

        if config is not None:
            cloned_pipeline.set_hyperparameters(config.get_dictionary())
        else:
            # Derive configuration on complete data set. Test performance via CV
            cloned_pipeline.cfg_cache = cfg_cache
            cloned_pipeline.cfg_keys = cfg_keys
            cloned_pipeline.fit(ds.X, ds.y, logger=process_logger)
            config = process_logger.get_config(cloned_pipeline)
            pipeline.set_hyperparameters(config.get_dictionary())

        score, _, _ = self._score(ds, pipeline)
        return score

    def _score(self, ds: Dataset, estimator: Union[EstimatorComponent, FlexiblePipeline], n_folds: int = 4):
        y = ds.y
        y_pred, y_prob = self._cross_val_predict(estimator, ds.X, y, cv=n_folds)

        score = [util.score(y, y_prob, y_pred, ds.metric)]
        # Unfortunately meta-learning base was calculated using binarized_logloss instead of logloss.
        # As it is not possible to recalculate everything, we also add rocauc as a result
        if ds.metric == 'logloss':
            score.append(util.score(y, y_prob, y_pred, 'rocauc'))
        else:
            score.append(score[0])

        return score, y_pred, y_prob

    def create_estimator(self, conf: dict):
        try:
            name = conf['algorithm']
            kwargs = conf.copy()
            del kwargs['algorithm']

            module_name = name.rpartition(".")[0]
            class_name = name.split(".")[-1]

            module = importlib.import_module(module_name)
            class_ = getattr(module, class_name)
            return class_(**kwargs)
        except Exception as ex:
            self.logger.error('Invalid name with config {}'.format(conf))
            raise ex

    def transform_dataset(self, ds: Dataset, config: Configuration, component: EstimatorComponent) \
            -> Tuple[np.ndarray, Optional[float]]:
        component.set_hyperparameters(config.get_dictionary())
        if is_classifier(component):
            score, y_pred, y_prob = self._score(ds, component)
            try:
                y_pred = y_pred.astype(float)
            except ValueError:
                pass
            X = np.hstack((ds.X, y_prob, np.reshape(y_pred, (-1, 1))))
        else:
            X = component.fit(ds.X, ds.y).transform(ds.X)
            score = None
        return X, score
