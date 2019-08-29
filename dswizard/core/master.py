import copy
import logging
import os
import threading
import time
from typing import Tuple, Optional, List

import math
from ConfigSpace import Configuration

from dswizard.core.base_bandit_learner import BanditLearner
from dswizard.core.dispatcher import LocalDispatcher, PyroDispatcher
from dswizard.core.logger import JsonResultLogger
from dswizard.core.model import CandidateId, Job, CandidateStructure
from dswizard.core.runhistory import RunHistory
from dswizard.core.worker import Worker


class Master:
    def __init__(self,
                 run_id: str,
                 bandit_learner: BanditLearner,
                 working_directory: str = '.',
                 job_queue_sizes: Tuple[int, int] = (-1, 0),
                 dynamic_queue_size: bool = True,
                 logger: logging.Logger = None,
                 result_logger: JsonResultLogger = None,

                 ping_interval: int = 60,
                 nameserver: str = '127.0.0.1',
                 nameserver_port: int = None,
                 host: str = None,
                 local_workers: List[Worker] = None,
                 ):
        """
        The Master class is responsible for the book keeping and to decide what to run next. Optimizers are
        instantiations of Master, that handle the important steps of deciding what configurations to run on what
        budget when.
        :param run_id: A unique identifier of that Hyperband run. Use, for example, the cluster's JobID when running
            multiple concurrent runs to separate them
        :param bandit_learner: A hyperparameter optimization procedure
        :param working_directory: The top level working directory accessible to all compute nodes(shared filesystem).
        :param job_queue_sizes: min and max size of the job queue. During the run, when the number of jobs in the queue
            reaches the min value, it will be filled up to the max size. Default: (0,1)
        :param dynamic_queue_size:  Whether or not to change the queue size based on the number of workers available.
            If true (default), the job_queue_sizes are relative to the current number of workers.
        :param logger: the logger to output some (more or less meaningful) information
        :param result_logger: a result logger that writes live results to disk

        :param ping_interval: number of seconds between pings to discover new nodes. Default is 60 seconds.
        :param nameserver: address of the Pyro4 nameserver
        :param nameserver_port: port of Pyro4 nameserver
        :param host: IP (or name that resolves to that) of the network interface to use
        :param local_workers: A list of local workers. If this parameter is not None, a LocalDispatcher will be used.
        """

        self.working_directory = working_directory
        os.makedirs(self.working_directory, exist_ok=True)

        if logger is None:
            self.logger = logging.getLogger('Master')
        else:
            self.logger = logger

        self.result_logger = result_logger

        self.time_ref: Optional[float] = None

        self.bandit_learner: BanditLearner = bandit_learner
        self.jobs = []

        self.num_running_jobs = 0
        self.job_queue_sizes = job_queue_sizes
        self.user_job_queue_sizes = job_queue_sizes
        self.dynamic_queue_size = dynamic_queue_size

        if job_queue_sizes[0] >= job_queue_sizes[1]:
            raise ValueError('The queue size range needs to be (min, max) with min<max!')

        # condition to synchronize the job_callback and the queue
        self.thread_cond = threading.Condition()

        self.config = {
            'time_ref': self.time_ref
        }

        if local_workers is not None:
            self.dispatcher = LocalDispatcher(local_workers, self.job_callback, queue_callback=self.adjust_queue_size,
                                              run_id=run_id)
        else:
            self.dispatcher = PyroDispatcher(self.job_callback, queue_callback=self.adjust_queue_size, run_id=run_id,
                                             ping_interval=ping_interval, nameserver=nameserver,
                                             nameserver_port=nameserver_port, host=host)

        self.dispatcher_thread = threading.Thread(target=self.dispatcher.run)
        self.dispatcher_thread.start()

    def shutdown(self, shutdown_workers: bool = False) -> None:
        self.logger.info('shutdown initiated, shutdown_workers = {}'.format(shutdown_workers))
        self.dispatcher.shutdown(shutdown_workers)
        self.dispatcher_thread.join()

    def run(self, min_n_workers: int = 1, iteration_kwargs: dict = None) -> RunHistory:
        """
        run optimization
        :param min_n_workers: minimum number of workers before starting the run
        :param iteration_kwargs: additional kwargs for the iteration class. Defaults to empty dictionary
        :return:
        """

        if iteration_kwargs is None:
            iteration_kwargs = {}

        self._wait_for_workers(min_n_workers)

        iteration_kwargs.update({'result_logger': self.result_logger})

        if self.time_ref is None:
            self.time_ref = time.time()
            self.config['time_ref'] = self.time_ref

            self.logger.info('starting run at {}'.format(time.strftime('%Y-%m-%dT%H:%M:%S%z',
                                                                       time.localtime(self.time_ref))))

        self.thread_cond.acquire()
        self._queue_wait()

        # while time_limit is not exhausted:
        #   structure, budget = structure_generator.get_next_structure()
        #   configspace = structure.configspace
        #
        #   incumbent, loss = bandit_learners.optimize(configspace, structure)
        #   Update score of selected structure with loss

        # Main hyperparamter optimization logic
        self.bandit_learner.optimize(self._submit_job, iteration_kwargs)

        self.thread_cond.release()
        self.logger.info('Finished run after {} seconds'.format(math.ceil(time.time() - self.time_ref)))

        return RunHistory([copy.deepcopy(i.data) for i in self.bandit_learner.iterations],
                          {**self.config, **self.bandit_learner.config})

    def _wait_for_workers(self, min_n_workers: int = 1) -> None:
        """
        helper function to hold execution until some workers are active
        :param min_n_workers: minimum number of workers present before the run starts
        :return:
        """

        self.logger.debug('wait_for_workers trying to get the condition')
        with self.thread_cond:
            while self.dispatcher.number_of_workers() < min_n_workers:
                self.logger.debug('only {} worker(s) available, waiting for at least {}.'.format(
                    self.dispatcher.number_of_workers(), min_n_workers))
                self.thread_cond.wait(1)
                self.dispatcher.trigger_discover_worker()

        self.logger.debug('Enough workers to start this run!')

    def _submit_job(self,
                    cid: CandidateId,
                    cs: CandidateStructure,
                    config: Configuration = None
                    ) -> None:
        """
        protected function to submit a new job to the dispatcher

        This function handles the actual submission in a (hopefully) thread save way
        """
        self.logger.debug('submitting job {} to dispatcher'.format(cid))
        with self.thread_cond:
            job = Job(cid, config, cs.pipeline, cs.budget, cs.timeout)

            self.dispatcher.submit_job(job)
            self.num_running_jobs += 1
        self._queue_wait()

    def _queue_wait(self) -> None:
        """
        helper function to wait for the queue to not overflow/underload it
        """

        if self.num_running_jobs >= self.job_queue_sizes[1]:
            while self.num_running_jobs > self.job_queue_sizes[0]:
                self.logger.debug('running jobs: {}, queue sizes: {} -> wait'.format(
                    self.num_running_jobs, self.job_queue_sizes))
                self.thread_cond.wait()

    def adjust_queue_size(self, number_of_workers: int = None) -> None:
        self.logger.debug('number of workers changed to {}'.format(number_of_workers))
        with self.thread_cond:
            if self.dynamic_queue_size:
                nw = self.dispatcher.number_of_workers() if number_of_workers is None else number_of_workers
                self.job_queue_sizes = (self.user_job_queue_sizes[0] + nw, self.user_job_queue_sizes[1] + nw)
                self.logger.debug('adjusted queue size to {}'.format(self.job_queue_sizes))
            self.thread_cond.notify_all()

    def job_callback(self, job: Job) -> None:
        """
        method to be called when a job has finished

        this will do some book keeping and call the user defined new_result_callback if one was specified
        :param job: Finished Job
        :return:
        """
        self.logger.debug('job_callback for {} started'.format(job.id))
        with self.thread_cond:
            self.num_running_jobs -= 1

            if self.result_logger is not None:
                self.result_logger.log_evaluated_config(job)

            self.bandit_learner.config_generator.register_result(job)

            if self.num_running_jobs <= self.job_queue_sizes[0]:
                self.logger.debug('Trying to start next job!')
                self.thread_cond.notify()
