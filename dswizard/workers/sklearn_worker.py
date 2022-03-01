import os
import warnings
from typing import Optional, Tuple, Union, List

import joblib
import numpy as np
from ConfigSpace import Configuration
from sklearn import clone
from sklearn.base import is_classifier
from sklearn.model_selection import check_cv, train_test_split
from sklearn.model_selection._validation import _fit_and_predict, _check_is_permutation
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples

from dswizard.components.base import EstimatorComponent
from dswizard.core.config_cache import ConfigCache
from dswizard.core.logger import ProcessLogger
from dswizard.core.model import CandidateId, ConfigKey, Dataset
from dswizard.core.worker import Worker
from dswizard.pipeline.pipeline import FlexiblePipeline
from dswizard.util import util
from dswizard.util.util import model_file

warnings.filterwarnings("ignore", category=UserWarning)


class SklearnWorker(Worker):

    def compute(self,
                ds: Dataset,
                cid: CandidateId,
                config: Optional[Configuration],
                cfg_cache: Optional[ConfigCache],
                cfg_keys: Optional[List[ConfigKey]],
                pipeline: FlexiblePipeline,
                process_logger: ProcessLogger) -> List[float]:
        if config is None:
            # Derive configuration on complete data set. Test performance via CV
            # noinspection PyTypeChecker
            cloned_pipeline: FlexiblePipeline = clone(pipeline)
            cloned_pipeline.cid = cid
            cloned_pipeline.cfg_cache = cfg_cache
            cloned_pipeline.cfg_keys = cfg_keys
            cloned_pipeline.fit(ds.X, ds.y, logger=process_logger)
            config = process_logger.get_config(cloned_pipeline)

        # noinspection PyTypeChecker
        cloned_pipeline: FlexiblePipeline = clone(pipeline)
        cloned_pipeline.set_hyperparameters(config.get_dictionary())
        score, _, _, models = self._score(ds, cloned_pipeline)
        self._store_models(cid, models)
        return score

    def transform_dataset(self, ds: Dataset, cid: CandidateId, component: EstimatorComponent,
                          config: Configuration) -> Tuple[np.ndarray, Optional[float]]:
        component.set_hyperparameters(config.get_dictionary())
        if is_classifier(component):
            score, y_pred, y_prob, models = self._score(ds, component)
            # TODO crude fix for holdout score. Fix this
            if y_pred.shape != ds.y.shape:
                y_pred = models[0].predict(ds.X)
                y_prob = models[0].predict_proba(ds.X)

            X = np.hstack((ds.X, y_prob, np.reshape(y_pred, (-1, 1))))
        else:
            models = [component.fit(ds.X, ds.y)]
            X = models[0].transform(ds.X)
            score = [None, None]
        self._store_models(cid, models)
        return X, score

    def _score(self, ds: Dataset, estimator: Union[EstimatorComponent, FlexiblePipeline], use_cv: bool = False) \
            -> Tuple[List[float], np.ndarray, np.ndarray, List[FlexiblePipeline]]:
        # TODO improve handling of holdout or cross-val prediction
        if use_cv:
            y, y_pred, y_prob, models = self._cross_val_predict(estimator, ds.X, ds.y, cv=4)
        else:
            y, y_pred, y_prob, models = self._holdout_predict(estimator, ds.X, ds.y)

        # Meta-learning only considers f1. Calculate f1 score for structure search
        score = [util.score(y, y_prob, y_pred, ds.metric), util.score(y, y_prob, y_pred, 'f1')]
        return score, y_pred, y_prob, models

    @staticmethod
    def _holdout_predict(pipeline, X, y=None, test_size=0.2) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[FlexiblePipeline]]:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
        cloned_pipeline: FlexiblePipeline = clone(pipeline)
        cloned_pipeline.fit(X_train, y_train)
        y_pred = cloned_pipeline.predict(X_test)
        y_prob = cloned_pipeline.predict_proba(X_test)
        return y_test, y_pred, y_prob, [cloned_pipeline]

    @staticmethod
    def _cross_val_predict(pipeline, X, y=None, cv=None) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[FlexiblePipeline]]:
        X, y, groups = indexable(X, y, None)
        cv = check_cv(cv, y, classifier=is_classifier(pipeline))
        cv.random_state = 42

        prediction_blocks = []
        probability_blocks = []
        fitted_pipelines = []
        for train, test in cv.split(X, y, groups):
            cloned_pipeline = clone(pipeline)
            probability_blocks.append(
                (_fit_and_predict(cloned_pipeline, X, y, train, test, 0, {}, 'predict_proba'), test)
            )
            prediction_blocks.append(cloned_pipeline.predict(X))
            fitted_pipelines.append(cloned_pipeline)

        # Concatenate the predictions
        probabilities = [prob_block_i for prob_block_i, _ in probability_blocks]
        predictions = [pred_block_i for pred_block_i in prediction_blocks]
        test_indices = np.concatenate([indices_i for _, indices_i in probability_blocks])

        if not _check_is_permutation(test_indices, _num_samples(X)):
            raise ValueError('cross_val_predict only works for partitions')

        inv_test_indices = np.empty(len(test_indices), dtype=int)
        inv_test_indices[test_indices] = np.arange(len(test_indices))

        probabilities = np.concatenate(probabilities)
        predictions = np.concatenate(predictions)

        if isinstance(predictions, list):
            return y, [p[inv_test_indices] for p in predictions], [p[inv_test_indices] for p in
                                                                   probabilities], fitted_pipelines
        else:
            return y, predictions[inv_test_indices], probabilities[inv_test_indices], fitted_pipelines

    def _store_models(self, cid: CandidateId, models: List[Union[EstimatorComponent, FlexiblePipeline]]):
        name = model_file(cid)
        file = os.path.join(self.workdir, name)
        with open(file, 'wb') as f:
            joblib.dump(models, f)
