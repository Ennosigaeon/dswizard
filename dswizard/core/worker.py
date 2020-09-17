from __future__ import annotations

import abc
import copy
import logging
import os
import socket
import threading
import timeit
from typing import Optional, TYPE_CHECKING, Tuple, List

import numpy as np
from ConfigSpace import Configuration
from sklearn.base import is_classifier
from sklearn.model_selection import check_cv
from sklearn.model_selection._validation import _fit_and_predict, _check_is_permutation
from sklearn.utils import indexable
from sklearn.utils.validation import _num_samples

import pynisher2
from automl.components.base import EstimatorComponent
from dswizard.components.pipeline import FlexiblePipeline
from dswizard.core.logger import ProcessLogger
from dswizard.core.model import Result, StatusType, Runtime, Dataset, Job

if TYPE_CHECKING:
    from dswizard.core.dispatcher import Dispatcher
    from dswizard.core.model import CandidateId
    from dswizard.core.config_cache import ConfigCache


class Worker(abc.ABC):
    """
    The worker is responsible for evaluating a single configuration on a single budget at a time. Communication to the
    individual workers goes via the nameserver, management of the worker-pool and job scheduling is done by the
    Dispatcher and jobs are determined by the Master. In distributed systems, each cluster-node runs a Worker-instance.
    To implement your own worker, overwrite the `__init__`- and the `compute`-method. The first allows to perform
    initial computations, e.g. loading the dataset, when the worker is started, while the latter is repeatedly called
    during the optimization and evaluates a given configuration yielding the associated loss.
    """

    def __init__(self,
                 run_id: str,
                 logger: logging.Logger = None,
                 wid: str = None,
                 metric: str = 'f1',
                 cfg_cache: Optional[ConfigCache] = None,
                 workdir: str = '/tmp/dswizzard/'):
        """
        :param run_id: unique id to identify individual optimization run
        :param logger: logger used for debugging output
        :param wid: if multiple workers are started in the same process, you MUST provide a unique id for each one of
            them using the `id` argument.
        :type metric: Allowed values are 'accuracy', 'precision', 'recall', 'f1' (default), 'logloss' and 'rocauc'
        """
        self.run_id = run_id
        self.worker_id = '{}.worker.{}'.format(self.run_id, os.getpid())
        self.metric = metric

        self.cfg_cache = cfg_cache

        self.workdir = workdir
        self.process_logger: ProcessLogger = None

        if wid is not None:
            self.worker_id += '.{}'.format(wid)

        if logger is None:
            self.logger = logging.getLogger(self.worker_id)
        else:
            self.logger = logger

        self.logger.info('Running on {} with pid {}'.format(socket.gethostname(), os.getpid()))

        self.start_time: float = None
        self.busy = False
        self.thread_cond = threading.Condition(threading.Lock())

    def start_computation(self,
                          callback: Dispatcher,
                          job: Job) -> Result:
        with self.thread_cond:
            while self.busy:
                self.thread_cond.wait()
            self.busy = True

        self.logger.info('start processing job {}'.format(job.cid))

        result = None
        try:
            self.process_logger = ProcessLogger(self.workdir, job.cid)
            wrapper = pynisher2.enforce_limits(wall_time_in_s=job.cutoff)(self.compute)
            c = wrapper(job.ds, job.cid, job.config, self.cfg_cache, job.cfg_keys, job.component, **job.kwargs)

            if wrapper.exit_status is pynisher2.TimeoutException:
                status = StatusType.TIMEOUT
                cost = 1
            elif wrapper.exit_status is pynisher2.MemorylimitException:
                status = StatusType.MEMOUT
                cost = 1
            elif wrapper.exit_status == 0 and c is not None:
                status = StatusType.SUCCESS
                cost = c
            else:
                status = StatusType.CRASHED
                self.logger.debug('Worker failed with {}\n{}'.format(c[0], c[1]))
                cost = 1
            runtime = Runtime(wrapper.wall_clock_time, timestamp=timeit.default_timer() - self.start_time)

            if job.config is None:
                config, partial_configs = self.process_logger.restore_config(job.component)
            else:
                config = job.config
                partial_configs = None

            # job.component has to be always a FlexiblePipeline
            steps = [(name, comp.name()) for name, comp in job.component.steps]
            result = Result(status, config, cost, runtime, partial_configs)
        except KeyboardInterrupt:
            raise
        except Exception as ex:
            # Should never occur, just a safety net
            self.logger.exception('Unexpected error during computation: \'{}\''.format(ex))
            result = Result(StatusType.CRASHED, config if 'config' in locals() else job.config, 1, None,
                            partial_configs if 'partial_configs' in locals() else None)
        finally:
            self.process_logger = None
            with self.thread_cond:
                self.busy = False
                callback.register_result(job.cid, result)
                self.thread_cond.notify()
        self.logger.debug('job {} finished with: {} -> {}'.format(job.cid, result.status, result.loss))
        return result

    @staticmethod
    def _cross_val_predict(pipeline, X, y=None, cv=None):
        X, y, groups = indexable(X, y, None)

        cv = check_cv(cv, y, classifier=is_classifier(pipeline))

        prediction_blocks = []
        for train, test in cv.split(X, y, groups):
            cloned_pipeline = copy.copy(pipeline)
            prediction_blocks.append(_fit_and_predict(cloned_pipeline, X, y, train, test, 0, {}, 'predict'))

        # Concatenate the predictions
        predictions = [pred_block_i for pred_block_i, _ in prediction_blocks]
        test_indices = np.concatenate([indices_i
                                       for _, indices_i in prediction_blocks])

        if not _check_is_permutation(test_indices, _num_samples(X)):
            raise ValueError('cross_val_predict only works for partitions')

        inv_test_indices = np.empty(len(test_indices), dtype=int)
        inv_test_indices[test_indices] = np.arange(len(test_indices))

        predictions = np.concatenate(predictions)

        if isinstance(predictions, list):
            return [p[inv_test_indices] for p in predictions]
        else:
            return predictions[inv_test_indices]

    @abc.abstractmethod
    def compute(self,
                ds: Dataset,
                config_id: CandidateId,
                config: Optional[Configuration],
                cfg_cache: Optional[ConfigCache],
                cfg_keys: Optional[List[Tuple[float, int]]],
                pipeline: FlexiblePipeline,
                budget: float
                ) -> float:
        """
        The function you have to overload implementing your computation.
        :param ds:
        :param config_id: the id of the configuration to be evaluated
        :param config: the actual configuration to be evaluated.
        :param cfg_cache:
        :param cfg_keys:
        :param pipeline: Additional information about the sampled configuration like pipeline structure.
        :param budget: the budget for the evaluation
        """
        pass

    def start_transform_dataset(self, job: Job) -> Result:
        self.logger.info('start processing job {} with estimator {}'.format(job.cid, job.cs))
        X = None
        try:
            wrapper = pynisher2.enforce_limits(wall_time_in_s=job.cutoff)(self.transform_dataset)
            X, score = wrapper(job.ds, job.config, job.component)

            if wrapper.exit_status == 0 and X is not None:
                status = StatusType.SUCCESS
            else:
                status = StatusType.CRASHED
        except KeyboardInterrupt:
            raise
        except Exception as ex:
            # Should never occur, just a safety net
            self.logger.exception('Unexpected error during computation: \'{}\''.format(ex))
            status = StatusType.CRASHED
            score = 1
        return Result(status=status, loss=score, transformed_X=X,
                      runtime=Runtime(wrapper.wall_clock_time, timeit.default_timer() - self.start_time))

    @abc.abstractmethod
    def transform_dataset(self,
                          ds: Dataset,
                          config: Configuration,
                          component: EstimatorComponent) -> Tuple[np.ndarray, Optional[float]]:
        pass

    def is_busy(self) -> bool:
        return self.busy
