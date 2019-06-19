import copy
import logging
import os
import threading
import time
from typing import Tuple, Optional, List, Any

from hpbandster.core.base_structure_generator import BaseStructureGenerator
from hpbandster.core.base_iteration import WarmStartIteration, BaseIteration
from hpbandster.core.dispatcher import Dispatcher
from hpbandster.core.model import ConfigId
from hpbandster.core.model import Job
from hpbandster.core.result import JsonResultLogger
from hpbandster.core.result import Result


class Master(object):
    def __init__(self,
                 run_id: str,
                 config_generator: BaseStructureGenerator,
                 working_directory: str = '.',
                 ping_interval: int = 60,
                 nameserver: str = '127.0.0.1',
                 nameserver_port: int = None,
                 host: str = None,
                 job_queue_sizes: Tuple[int, int] = (-1, 0),
                 dynamic_queue_size: bool = True,
                 logger: logging.Logger = None,
                 result_logger: JsonResultLogger = None,
                 previous_result: Any = None,
                 ):
        """
        The Master class is responsible for the book keeping and to decide what to run next. Optimizers are
        instantiations of Master, that handle the important steps of deciding what configurations to run on what
        budget when.
        :param run_id: A unique identifier of that Hyperband run. Use, for example, the cluster's JobID when running
            multiple concurrent runs to separate them
        :param config_generator: An object that can generate new configurations and registers results of executed runs
        :param working_directory: The top level working directory accessible to all compute nodes(shared filesystem).
        :param ping_interval: number of seconds between pings to discover new nodes. Default is 60 seconds.
        :param nameserver: address of the Pyro4 nameserver
        :param nameserver_port: port of Pyro4 nameserver
        :param host: IP (or name that resolves to that) of the network interface to use
        :param job_queue_sizes: min and max size of the job queue. During the run, when the number of jobs in the queue
            reaches the min value, it will be filled up to the max size. Default: (0,1)
        :param dynamic_queue_size:  Whether or not to change the queue size based on the number of workers available.
            If true (default), the job_queue_sizes are relative to the current number of workers.
        :param logger: the logger to output some (more or less meaningful) information
        :param result_logger: a result logger that writes live results to disk
        :param previous_result: previous run to warmstart the run
        """

        self.working_directory = working_directory
        os.makedirs(self.working_directory, exist_ok=True)

        if logger is None:
            self.logger = logging.getLogger('hpbandster')
        else:
            self.logger = logger

        self.result_logger = result_logger

        self.config_generator = config_generator
        self.time_ref: Optional[float] = None

        self.iterations: List[BaseIteration] = []
        self.jobs = []

        self.num_running_jobs = 0
        self.job_queue_sizes = job_queue_sizes
        self.user_job_queue_sizes = job_queue_sizes
        self.dynamic_queue_size = dynamic_queue_size

        if job_queue_sizes[0] >= job_queue_sizes[1]:
            raise ValueError('The queue size range needs to be (min, max) with min<max!')

        if previous_result is None:
            self.warmstart_iteration = []

        else:
            self.warmstart_iteration = [WarmStartIteration(previous_result, self.config_generator)]

        # condition to synchronize the job_callback and the queue
        self.thread_cond = threading.Condition()

        self.config = {
            'time_ref': self.time_ref
        }

        self.dispatcher = Dispatcher(self.job_callback, queue_callback=self.adjust_queue_size, run_id=run_id,
                                     ping_interval=ping_interval, nameserver=nameserver,
                                     nameserver_port=nameserver_port, host=host)

        self.dispatcher_thread = threading.Thread(target=self.dispatcher.run)
        self.dispatcher_thread.start()

    def shutdown(self, shutdown_workers: bool = False) -> None:
        self.logger.info('HBMASTER: shutdown initiated, shutdown_workers = {}'.format(shutdown_workers))
        self.dispatcher.shutdown(shutdown_workers)
        self.dispatcher_thread.join()

    def wait_for_workers(self, min_n_workers: int = 1) -> None:
        """
        helper function to hold execution until some workers are active
        :param min_n_workers: minimum number of workers present before the run starts
        :return:
        """

        self.logger.debug('wait_for_workers trying to get the condition')
        with self.thread_cond:
            while self.dispatcher.number_of_workers() < min_n_workers:
                self.logger.debug('HBMASTER: only {} worker(s) available, waiting for at least {}.'.format(
                    self.dispatcher.number_of_workers(), min_n_workers))
                self.thread_cond.wait(1)
                self.dispatcher.trigger_discover_worker()

        self.logger.debug('Enough workers to start this run!')

    def get_next_iteration(self, iteration: int, iteration_kwargs: dict) -> BaseIteration:
        """
        instantiates the next iteration

        Overwrite this to change the iterations for different optimizers
        :param iteration: the index of the iteration to be instantiated
        :param iteration_kwargs: additional kwargs for the iteration class. Defaults to empty dictionary
        :return: a valid HB iteration object
        """

        raise NotImplementedError('implement get_next_iteration for {}'.format(type(self).__name__))

    def run(self, n_iterations: int = 1, min_n_workers: int = 1, iteration_kwargs: dict = None) -> Result:
        """
        run n_iterations of SuccessiveHalving
        :param n_iterations: number of iterations to be performed in this run
        :param min_n_workers: minimum number of workers before starting the run
        :param iteration_kwargs: additional kwargs for the iteration class. Defaults to empty dictionary
        :return:
        """

        if iteration_kwargs is None:
            iteration_kwargs = {}

        self.wait_for_workers(min_n_workers)

        iteration_kwargs.update({'result_logger': self.result_logger})

        if self.time_ref is None:
            self.time_ref = time.time()
            self.config['time_ref'] = self.time_ref

            self.logger.info('HBMASTER: starting run at {}'.format(self.time_ref))

        self.thread_cond.acquire()
        while True:

            self._queue_wait()

            next_run = None
            # find a new run to schedule
            for i in self.active_iterations():
                next_run = self.iterations[i].get_next_run()
                if next_run is not None:
                    break

            if next_run is not None:
                # noinspection PyUnboundLocalVariable
                self.logger.debug('HBMASTER: schedule new run for iteration {}'.format(i))
                self._submit_job(*next_run)
                continue
            else:
                if n_iterations > 0:  # we might be able to start the next iteration
                    self.iterations.append(self.get_next_iteration(len(self.iterations), iteration_kwargs))
                    n_iterations -= 1
                    continue

            # at this point there is no immediate run that can be scheduled,
            # so wait for some job to finish if there are active iterations
            if self.active_iterations():
                self.thread_cond.wait()
            else:
                break

        self.thread_cond.release()

        for i in self.warmstart_iteration:
            i.fix_timestamps(self.time_ref)

        ws_data = [i.data for i in self.warmstart_iteration]

        return Result([copy.deepcopy(i.data) for i in self.iterations] + ws_data, self.config)

    def adjust_queue_size(self, number_of_workers: int = None) -> None:
        self.logger.debug('HBMASTER: number of workers changed to {}'.format(number_of_workers))
        with self.thread_cond:
            self.logger.debug('adjust_queue_size: lock acquired')
            if self.dynamic_queue_size:
                nw = self.dispatcher.number_of_workers() if number_of_workers is None else number_of_workers
                self.job_queue_sizes = (self.user_job_queue_sizes[0] + nw, self.user_job_queue_sizes[1] + nw)
                self.logger.info('HBMASTER: adjusted queue size to {}'.format(self.job_queue_sizes))
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
            self.logger.debug('job_callback for {} got condition'.format(job.id))
            self.num_running_jobs -= 1

            if self.result_logger is not None:
                self.result_logger(job)
            self.iterations[job.id.iteration].register_result(job)
            self.config_generator.new_result(job)

            if self.num_running_jobs <= self.job_queue_sizes[0]:
                self.logger.debug('HBMASTER: Trying to run another job!')
                self.thread_cond.notify()

        self.logger.debug('job_callback for {} finished'.format(job.id))

    def _queue_wait(self) -> None:
        """
        helper function to wait for the queue to not overflow/underload it
        """

        if self.num_running_jobs >= self.job_queue_sizes[1]:
            while self.num_running_jobs > self.job_queue_sizes[0]:
                self.logger.debug('HBMASTER: running jobs: {}, queue sizes: {} -> wait'.format(
                    self.num_running_jobs, self.job_queue_sizes))
                self.thread_cond.wait()

    def _submit_job(self, config_id: ConfigId, config: dict, budget: float, config_info: dict) -> None:
        """
        protected function to submit a new job to the dispatcher

        This function handles the actual submission in a (hopefully) thread save way
        """
        self.logger.debug('HBMASTER: trying submitting job {} to dispatcher'.format(config_id))
        with self.thread_cond:
            self.logger.debug('HBMASTER: submitting job {} to dispatcher'.format(config_id))
            self.dispatcher.submit_job(config_id, config=config, config_info=config_info,
                                       budget=budget, working_directory=self.working_directory)
            self.num_running_jobs += 1
            self.logger.debug('HBMASTER: job {} submitted to dispatcher'.format(config_id))

    def active_iterations(self) -> List[int]:
        """
        function to find active (not marked as finished) iterations
        :return: all active iteration indices (empty if there are none)
        """

        return list(filter(lambda idx: not self.iterations[idx].is_finished, range(len(self.iterations))))
