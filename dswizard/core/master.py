from __future__ import annotations

import copy
import logging
import multiprocessing
import os
import threading
import time
from multiprocessing.managers import SyncManager
from typing import Type, TYPE_CHECKING

import math
from ConfigSpace.configuration_space import Configuration, ConfigurationSpace

from core.dispatcher import Dispatcher
from dswizard.core.config_cache import ConfigCache
from dswizard.core.model import Job, Dataset
from dswizard.core.runhistory import RunHistory
from dswizard.optimizers.bandit_learners import HyperbandLearner
from dswizard.optimizers.config_generators import RandomSampling
from workers import SklearnWorker

if TYPE_CHECKING:
    from dswizard.core.base_bandit_learner import BanditLearner
    from dswizard.core.base_config_generator import BaseConfigGenerator
    from dswizard.core.logger import JsonResultLogger
    from dswizard.core.worker import Worker


class Master:
    def __init__(self,
                 run_id: str,
                 working_directory: str = '.',
                 logger: logging.Logger = None,
                 result_logger: JsonResultLogger = None,

                 n_workers: int = 1,
                 worker_class: Type[Worker] = SklearnWorker,

                 config_generator_class: Type[BaseConfigGenerator] = RandomSampling,
                 config_generator_kwargs: dict = None,

                 bandit_learner_class: Type[BanditLearner] = HyperbandLearner,
                 bandit_learner_kwargs: dict = None
                 ):
        """
        The Master class is responsible for the book keeping and to decide what to run next. Optimizers are
        instantiations of Master, that handle the important steps of deciding what configurations to run on what
        budget when.
        :param run_id: A unique identifier of that Hyperband run. Use, for example, the cluster's JobID when running
            multiple concurrent runs to separate them
        :param working_directory: The top level working directory accessible to all compute nodes(shared filesystem).
        :param logger: the logger to output some (more or less meaningful) information
        :param result_logger: a result logger that writes live results to disk
        """

        if bandit_learner_kwargs is None:
            bandit_learner_kwargs = {}
        if config_generator_kwargs is None:
            config_generator_kwargs = {}

        self.working_directory = os.path.join(working_directory, run_id)
        os.makedirs(self.working_directory, exist_ok=True)

        if 'working_directory' not in config_generator_kwargs:
            config_generator_kwargs['working_directory'] = self.working_directory

        if logger is None:
            self.logger = logging.getLogger('Master')
        else:
            self.logger = logger

        self.result_logger = result_logger
        self.jobs = []
        self.meta_data = {}

        # condition to synchronize the job_callback and the queue
        self.thread_cond = threading.Condition()

        SyncManager.register('ConfigCache', ConfigCache)
        mgr = multiprocessing.Manager()
        # noinspection PyUnresolvedReferences
        self.cfg_cache: ConfigCache = mgr.ConfigCache(clazz=config_generator_class,
                                                      init_kwargs=config_generator_kwargs,
                                                      run_id=run_id)

        if n_workers < 1:
            raise ValueError('Expected at least 1 worker, given {}'.format(n_workers))
        worker = []
        for i in range(n_workers):
            worker.append(worker_class(run_id=run_id, wid=str(i), cfg_cache=self.cfg_cache,
                                       workdir=self.working_directory))
        self.dispatcher = Dispatcher(worker, self.job_callback, run_id=run_id)

        self.bandit_learner: BanditLearner = bandit_learner_class(run_id=run_id, **bandit_learner_kwargs)

        self.dispatcher_thread = threading.Thread(target=self.dispatcher.run)
        self.dispatcher_thread.start()

    def shutdown(self) -> None:
        self.logger.info('shutdown initiated')
        self.dispatcher.shutdown()
        self.dispatcher_thread.join()

    def optimize(self, ds: Dataset,
                 n_configs: int = 1,
                 timeout: int = None,
                 pre_sample: bool = True) -> RunHistory:
        """
        run optimization
        :param ds:
        :param n_configs:
        :param timeout:
        :param pre_sample:
        :return:
        """

        start = time.time()
        self.meta_data['start'] = start
        self.logger.info('starting run at {}'.format(time.strftime('%Y-%m-%dT%H:%M:%S%z',
                                                                   time.localtime(start))))

        # while time_limit is not exhausted:
        #   structure, budget = structure_generator.get_next_structure()
        #   configspace = structure.configspace
        #
        #   incumbent, loss = bandit_learners.optimize(configspace, structure)
        #   Update score of selected structure with loss

        # Main hyperparamter optimization logic
        for candidate, iteration in self.bandit_learner.next_candidate(ds.meta_features, {'timeout': timeout}):
            # Optimize hyperparameters
            for i in range(n_configs):
                config_id = candidate.cid.with_config(i)
                if pre_sample:
                    config, cfg_key = self.cfg_cache.sample_configuration(candidate.budget,
                                                                          candidate.pipeline.configuration_space,
                                                                          ds.meta_features)
                    job = Job(ds, config_id, candidate, config, cfg_key)
                else:
                    job = Job(ds, config_id, candidate, None)
                self.dispatcher.submit_job(job)

        end = time.time()
        self.meta_data['end'] = end
        self.logger.info('Finished run after {} seconds'.format(math.ceil(end - start)))

        # TODO runhistory does not contain useful information yet
        return RunHistory([copy.deepcopy(i.data) for i in self.bandit_learner.iterations],
                          {**self.meta_data, **self.bandit_learner.meta_data})

    def job_callback(self, job: Job) -> None:
        """
        method to be called when a job has finished

        this will do some book keeping and call the user defined new_result_callback if one was specified
        :param job: Finished Job
        :return:
        """
        with self.thread_cond:
            try:
                if job.config is None:
                    self.logger.error(
                        'Encountered job without a configuration: {}. Using empty config as fallback'.format(job.cid))
                    job.config = Configuration(ConfigurationSpace())

                if self.result_logger is not None:
                    self.result_logger.log_evaluated_config(job)

                job.cs.add_result(job.result)
                self.cfg_cache.register_result(job)
                self.bandit_learner.register_result(job)
            except KeyboardInterrupt:
                raise
            except Exception as ex:
                self.logger.fatal('Encountered unhandled exception {}. This should never happen!'.format(ex),
                                  exc_info=True)
