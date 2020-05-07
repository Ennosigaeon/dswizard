import importlib
import timeit
import warnings
from typing import Optional, Tuple

import math
from ConfigSpace import Configuration
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_predict

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
                pipeline: FlexiblePipeline,
                budget: float,
                n_folds: int = 4) -> Tuple[float, Runtime]:
        start = timeit.default_timer()

        # Only use budget-percent
        n = math.ceil(len(ds.X) * budget)
        X = ds.X[:n]
        y = ds.y[:n]

        if config is not None:
            pipeline.set_hyperparameters(config.get_dictionary())
            pipeline.fit(X, y, budget=budget)
        else:
            pipeline.cfg_cache = cfg_cache
            pipeline.fit(X, y, budget=budget, logger=self.process_logger)

        try:
            y_pred = cross_val_predict(pipeline, X, y, cv=min(n_folds, n))
        except Exception as ex:
            self.logger.exception(ex)
            raise ex

        if self.metric == 'accuracy':
            score = 1 - accuracy_score(y, y_pred)
        elif self.metric == 'precision':
            score = 1 - precision_score(y, y_pred, average='weighted')
        elif self.metric == 'recall':
            score = 1 - recall_score(y, y_pred, average='weighted')
        elif self.metric == 'f1':
            score = 1 - f1_score(y, y_pred, average='weighted')
        elif self.metric == 'logloss':
            # TODO not working
            score = logloss(y, y_pred)
        elif self.metric == 'rocauc':
            score = 1 - multiclass_roc_auc_score(y, y_pred, average='weighted')
        else:
            raise ValueError

        # Always compute minimization problem
        return score, Runtime(timeit.default_timer() - start, pipeline.fit_time, pipeline.config_time)

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
            self.logger.error('Invalid estimator with config {}'.format(conf))
            raise ex
