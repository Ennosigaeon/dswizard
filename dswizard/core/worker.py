from __future__ import annotations

import abc
import logging
import os
import socket
import timeit
from typing import Optional, TYPE_CHECKING, Tuple, List

import numpy as np
from ConfigSpace import Configuration

from dswizard import pynisher2 as pynisher
from dswizard.components.base import EstimatorComponent
from dswizard.core.logger import ProcessLogger
from dswizard.core.model import Result, StatusType, Runtime, Dataset, EvaluationJob
from dswizard.pipeline.pipeline import FlexiblePipeline
from dswizard.util import util

if TYPE_CHECKING:
    from dswizard.core.model import CandidateId, ConfigKey
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
                 logger: logging.Logger = None,
                 wid: str = None,
                 cfg_cache: Optional[ConfigCache] = None,
                 workdir: str = '/tmp/dswizard/'):
        """
        :param logger: logger used for debugging output
        :param wid: if multiple workers are started in the same process, you MUST provide a unique id for each one of
            them using the `id` argument.
        """
        self.cfg_cache = cfg_cache
        self.workdir = workdir
        self.worker_id = f'worker.{wid}'

        if logger is None:
            self.logger = logging.getLogger('Worker')
        else:
            self.logger = logger

        self.logger.info(f'Running on {socket.gethostname()} with pid {os.getpid()}')

        self.start_time: Optional[float] = None
        self.busy = False

    def start_computation(self, job: EvaluationJob) -> Result:
        result = None
        try:
            process_logger = ProcessLogger(self.workdir, job.cid)
            wrapper = pynisher.enforce_limits(wall_time_in_s=job.cutoff, grace_period_in_s=5, logger=self.logger)(
                self.compute)
            c = wrapper(job.ds, job.cid, job.config, self.cfg_cache, job.cfg_keys, job.component, process_logger)

            if wrapper.exit_status is pynisher.TimeoutException:
                status = StatusType.TIMEOUT
                cost = util.worst_score(job.ds.metric)
            elif wrapper.exit_status is pynisher.MemorylimitException:
                status = StatusType.MEMOUT
                cost = util.worst_score(job.ds.metric)
            elif wrapper.exit_status == 0 and c is not None:
                status = StatusType.SUCCESS
                cost = c
            else:
                status = StatusType.CRASHED
                self.logger.debug(f'Worker failed with {c[0] if isinstance(c, Tuple) else c}')
                cost = util.worst_score(job.ds.metric)
            runtime = Runtime(wrapper.wall_clock_time, timestamp=timeit.default_timer() - self.start_time)

            if job.config is None:
                config, partial_configs = process_logger.restore_config(job.component)
            else:
                config = job.config
                partial_configs = None

            # job.component has to be always a FlexiblePipeline
            steps = [(name, comp.name()) for name, comp in job.component.steps]
            result = Result(job.cid, status, config, cost[0], cost[1], runtime, partial_configs)
        except KeyboardInterrupt:
            raise
        except Exception as ex:
            # Should never occur, just a safety net
            self.logger.exception(f'Unexpected error during computation: \'{ex}\'')
            # noinspection PyUnboundLocalVariable
            result = Result(job.cid, StatusType.CRASHED, config if 'config' in locals() else job.config,
                            util.worst_score(job.ds.metric)[0], util.worst_score(job.ds.metric)[0], None,
                            partial_configs if 'partial_configs' in locals() else None)
        return result

    @abc.abstractmethod
    def compute(self,
                ds: Dataset,
                cid: CandidateId,
                config: Optional[Configuration],
                cfg_cache: Optional[ConfigCache],
                cfg_keys: Optional[List[ConfigKey]],
                pipeline: FlexiblePipeline,
                process_logger: ProcessLogger
                ) -> List[float]:
        """
        The function you have to overload implementing your computation.
        :param ds:
        :param cid: the id of the configuration to be evaluated
        :param config: the actual configuration to be evaluated.
        :param cfg_cache:
        :param cfg_keys:
        :param pipeline: Additional information about the sampled configuration like pipeline structure.
        :param process_logger:
        """
        pass

    def start_transform_dataset(self, job: EvaluationJob) -> Result:
        self.logger.info(f'start transforming job {job.cid}')

        X = None
        try:
            wrapper = pynisher.enforce_limits(wall_time_in_s=job.cutoff, grace_period_in_s=5, logger=self.logger)(
                self.transform_dataset)
            c = wrapper(job.ds, job.cid, job.component, job.config)

            if wrapper.exit_status is pynisher.TimeoutException:
                status = StatusType.TIMEOUT
                score = util.worst_score(job.ds.metric)
            elif wrapper.exit_status is pynisher.MemorylimitException:
                status = StatusType.MEMOUT
                score = util.worst_score(job.ds.metric)
            elif wrapper.exit_status == 0 and c is not None:
                status = StatusType.SUCCESS
                X, score = c
            else:
                status = StatusType.CRASHED
                self.logger.debug(f'Worker failed with {c[0] if isinstance(c, Tuple) else c}')
                score = util.worst_score(job.ds.metric)
            result = Result(job.cid, status=status, loss=score[0], structure_loss=score[1], transformed_X=X,
                            runtime=Runtime(wrapper.wall_clock_time, timeit.default_timer() - self.start_time))
        except KeyboardInterrupt:
            raise
        except Exception as ex:
            # Should never occur, just a safety net
            self.logger.exception(f'Unexpected error during computation: \'{ex}\'')
            result = Result(job.cid, status=StatusType.CRASHED, loss=util.worst_score(job.ds.metric)[0],
                            structure_loss=util.worst_score(job.ds.metric)[1])
        return result

    @abc.abstractmethod
    def transform_dataset(self,
                          ds: Dataset,
                          cid: CandidateId,
                          component: EstimatorComponent,
                          config: Configuration) -> Tuple[np.ndarray, Optional[float]]:
        pass
