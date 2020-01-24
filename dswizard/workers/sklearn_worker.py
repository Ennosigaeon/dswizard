import importlib
import timeit
from typing import Optional, Tuple

import math
from ConfigSpace import Configuration
from sklearn import metrics

from dswizard.components.pipeline import FlexiblePipeline
from dswizard.core.config_cache import ConfigCache
from dswizard.core.model import CandidateId, Runtime, Dataset
from dswizard.core.worker import Worker


class SklearnWorker(Worker):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def compute(self,
                ds: Dataset,
                config_id: CandidateId,
                config: Optional[Configuration],
                cfg_cache: Optional[ConfigCache],
                pipeline: FlexiblePipeline,
                budget: float) -> Tuple[float, Runtime]:
        start = timeit.default_timer()

        # Only use budget-percent
        # TODO use cross-validation
        # TODO shuffle dataset before truncating
        n = math.ceil(len(ds.X) * budget)

        # TODO for iris dataset not reasonable
        n = len(ds.X)

        X = ds.X[:n]
        y = ds.y[:n]

        self.logger.debug('starting to fit pipeline and predict values')

        if config is not None:
            pipeline.set_hyperparameters(config.get_dictionary())
            pipeline.fit(X, y, budget=budget)
        else:
            pipeline.cfg_cache = cfg_cache
            pipeline.fit(X, y, budget=budget, logger=self.process_logger)

        y_pred = pipeline.predict(ds.X_test)

        # TODO make metric flexible
        accuracy = metrics.accuracy_score(ds.y_test, y_pred)

        return 1 - accuracy, Runtime(timeit.default_timer() - start, pipeline.fit_time, pipeline.config_time)

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
