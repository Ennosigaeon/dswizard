from __future__ import annotations

import copy
import logging
import multiprocessing
import os
import threading
import time
from multiprocessing.managers import SyncManager
from typing import Optional, List, Type, TYPE_CHECKING

import math

from core.dispatcher import Dispatcher
from dswizard import utils
from dswizard.core.config_cache import ConfigCache
from dswizard.core.model import Job, Dataset
from dswizard.core.runhistory import RunHistory
from dswizard.optimizers.bandit_learners import HyperbandLearner
from dswizard.optimizers.config_generators import RandomSampling

if TYPE_CHECKING:
    from ConfigSpace.configuration_space import ConfigurationSpace
    from dswizard.core.base_bandit_learner import BanditLearner
    from dswizard.core.base_config_generator import BaseConfigGenerator
    from dswizard.core.logger import JsonResultLogger
    from dswizard.core.model import MetaFeatures
    from dswizard.core.worker import Worker


class Master:
    def __init__(self,
                 run_id: str,
                 working_directory: str = '.',
                 logger: logging.Logger = None,
                 result_logger: JsonResultLogger = None,

                 workers: List[Worker] = None,

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
            If true (default), the job_queue_sizes are relative to the current number of workers.
        :param logger: the logger to output some (more or less meaningful) information
        :param result_logger: a result logger that writes live results to disk

        :param workers: A list of local workers. If this parameter is not None, a LocalDispatcher will be used.
        """

        if bandit_learner_kwargs is None:
            bandit_learner_kwargs = {}
        if config_generator_kwargs is None:
            config_generator_kwargs = {}

        self.working_directory = working_directory
        os.makedirs(self.working_directory, exist_ok=True)

        if logger is None:
            self.logger = logging.getLogger('Master')
        else:
            self.logger = logger

        self.result_logger = result_logger

        self.jobs = []

        # condition to synchronize the job_callback and the queue
        self.thread_cond = threading.Condition()

        self.meta_data = {}

        self.dispatcher = Dispatcher(workers, self.job_callback, run_id=run_id)

        # TODO Only quick and dirty. Fix this!
        SyncManager.register('ConfigCache', ConfigCache)
        mgr = multiprocessing.Manager()

        utils._cfg_cache_instance = mgr.ConfigCache(clazz=config_generator_class,
                                                    init_kwargs=config_generator_kwargs,
                                                    run_id=run_id)

        self.cfg_cache: ConfigCache = utils._cfg_cache_instance
        self.bandit_learner: BanditLearner = bandit_learner_class(run_id=run_id, **bandit_learner_kwargs)

        self.dispatcher_thread = threading.Thread(target=self.dispatcher.run)
        self.dispatcher_thread.start()

    def shutdown(self) -> None:
        self.logger.info('shutdown initiated')
        self.dispatcher.shutdown()
        self.dispatcher_thread.join()

    def optimize(self, ds: Dataset,
                 iterations: int = 1,
                 sample_config: bool = True) -> RunHistory:
        """
        run optimization
        :param sample_config:
        :param iterations:
        :param ds:
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
        for candidate, iteration in self.bandit_learner.next_candidate():
            # Optimize hyperparameters
            for i in range(iterations):
                config_id = candidate.id.with_config(i)
                if sample_config:
                    cg = self._get_config_generator(candidate.budget, candidate.pipeline.configuration_space,
                                                    ds.meta_features)
                    config = cg.sample_config()
                    job = Job(ds, config_id, candidate, config)
                else:
                    job = Job(ds, config_id, candidate, None)
                self.dispatcher.submit_job(job)

        end = time.time()
        self.meta_data['end'] = end
        self.logger.info('Finished run after {} seconds'.format(math.ceil(end - start)))

        # TODO runhistory does not contain useful information yet
        return RunHistory([copy.deepcopy(i.data) for i in self.bandit_learner.iterations],
                          {**self.meta_data, **self.bandit_learner.meta_data})

    def _get_config_generator(self, budget: float, configspace: ConfigurationSpace, meta_features: MetaFeatures) -> \
            Optional[BaseConfigGenerator]:
        # TODO refractor
        cache = utils.get_config_generator_cache()
        return cache.get_config_generator(budget, configspace, meta_features)

    def job_callback(self, job: Job) -> None:
        """
        method to be called when a job has finished

        this will do some book keeping and call the user defined new_result_callback if one was specified
        :param job: Finished Job
        :return:
        """
        with self.thread_cond:
            if job.config is None:
                raise ValueError('Encountered job without a configuration: {}'.format(job))

            if self.result_logger is not None:
                self.result_logger.log_evaluated_config(job)

            job.cs.add_result(job.result)
            self.cfg_cache.register_result(job)
            self.bandit_learner.register_result(job)
