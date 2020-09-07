import importlib
import timeit
import warnings
from typing import Optional, Tuple, Union, List

import numpy as np
from ConfigSpace import Configuration
from sklearn import clone
from sklearn.base import is_classifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from automl.components.base import EstimatorComponent
from dswizard.components.pipeline import FlexiblePipeline
from dswizard.core.config_cache import ConfigCache
from dswizard.core.model import CandidateId, Runtime, Dataset
from dswizard.core.worker import Worker
from util.util import multiclass_roc_auc_score, logloss

warnings.filterwarnings("ignore", category=UserWarning)


class SklearnWorker(Worker):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute(self,
                ds: Dataset,
                config_id: CandidateId,
                config: Optional[Configuration],
                cfg_cache: Optional[ConfigCache],
                cfg_keys: Optional[List[Tuple[float, int]]],
                pipeline: FlexiblePipeline,
                **kwargs) -> Tuple[float, Runtime]:
        start = timeit.default_timer()

        if config is not None:
            pipeline.set_hyperparameters(config.get_dictionary())
        else:
            # Derive configuration on complete data set. Test performance via CV
            cloned_pipeline = clone(pipeline)
            cloned_pipeline.cfg_cache = cfg_cache
            cloned_pipeline.cfg_keys = cfg_keys
            cloned_pipeline.fit(ds.X, ds.y, logger=self.process_logger)
            config = self.process_logger.get_config(cloned_pipeline)
            pipeline.set_hyperparameters(config.get_dictionary())

        score, _ = self._score(ds, pipeline)
        return score, Runtime(timeit.default_timer() - start, pipeline.fit_time, pipeline.config_time)

    def _score(self, ds: Dataset, estimator: Union[EstimatorComponent, FlexiblePipeline], n_folds: int = 4):
        y = ds.y
        y_pred = self._cross_val_predict(estimator, ds.X, y, cv=n_folds)

        # Always compute minimization problem
        if self.metric == 'accuracy':
            score = -1 * accuracy_score(y, y_pred)
        elif self.metric == 'precision':
            score = -1 * precision_score(y, y_pred, average='weighted')
        elif self.metric == 'recall':
            score = -1 * recall_score(y, y_pred, average='weighted')
        elif self.metric == 'f1':
            score = -1 * f1_score(y, y_pred, average='weighted')
        elif self.metric == 'logloss':
            # TODO not working
            score = logloss(y, y_pred)
        elif self.metric == 'rocauc':
            score = -1 * multiclass_roc_auc_score(y, y_pred, average='weighted')
        else:
            raise ValueError
        return score, y_pred

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
            score, y_pred = self._score(ds, component)
            X = np.hstack((ds.X, np.reshape(y_pred, (-1, 1))))
        else:
            score = None
            X = component.fit(ds.X, ds.y).transform(ds.X)
        return X, score
