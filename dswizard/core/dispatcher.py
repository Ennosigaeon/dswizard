import logging
import queue
import threading
import time
from typing import Callable, Dict, Set, Optional

import Pyro4
from ConfigSpace import Configuration
from Pyro4.errors import ConnectionClosedError

from dswizard.core.model import ConfigId, Job, Result, ConfigInfo


class Worker:
    def __init__(self, name: str, uri: str):
        self.name = name
        self.proxy = Pyro4.Proxy(uri)
        self.runs_job: Optional[Job] = None

    def is_alive(self) -> bool:
        try:
            # noinspection PyProtectedMember
            self.proxy._pyroReconnect(1)
        except ConnectionClosedError:
            return False
        return True

    def shutdown(self) -> None:
        self.proxy.shutdown()

    def is_busy(self) -> bool:
        return self.proxy.is_busy()

    def __repr__(self):
        return self.name


class Dispatcher:
    """
    The dispatcher is responsible for assigning tasks to free workers, report results back to the master and
    communicate to the nameserver.
    """

    def __init__(self,
                 new_result_callback: Callable[[Job], None],
                 run_id: str = '0',
                 ping_interval: int = 10,
                 nameserver: str = 'localhost',
                 nameserver_port: int = None,
                 host: str = None,
                 logger: logging.Logger = None,
                 queue_callback: Callable[[int], None] = None):
        """
        :param new_result_callback: function that will be called with a `Job instance <dswizard.core.dispatcher.Job>`
            as argument. From the `Job` the result can be read and e.g. logged.
        :param run_id: unique run_id associated with the HPB run
        :param ping_interval: how often to ping for workers (in seconds)
        :param nameserver: address of the Pyro4 nameserver
        :param nameserver_port: port of Pyro4 nameserver
        :param host: ip (or name that resolves to that) of the network interface to use
        :param logger: logger-instance for info and debug
        :param queue_callback: gets called with the number of workers in the pool on every update-cycle
        """

        self.new_result_callback = new_result_callback
        self.queue_callback = queue_callback
        self.run_id = run_id
        self.nameserver = nameserver
        self.nameserver_port = nameserver_port
        self.host = host
        self.ping_interval = int(ping_interval)
        self.shutdown_all_threads = False

        if logger is None:
            self.logger = logging.getLogger('Dispatcher')
        else:
            self.logger = logger

        self.worker_pool: Dict[str, Worker] = {}

        self.waiting_jobs: queue.Queue[Job] = queue.Queue()
        self.running_jobs: Dict[ConfigId, Job] = {}
        self.idle_workers: Set[str] = set()

        self.thread_lock = threading.Lock()
        self.runner_cond = threading.Condition(self.thread_lock)
        self.discover_cond = threading.Condition(self.thread_lock)

        self.pyro_id = '{}.dispatcher'.format(self.run_id)
        self.pyro_daemon = None

    def run(self) -> None:
        with self.discover_cond:
            t1 = threading.Thread(target=self.discover_workers, name='discover_workers')
            t1.start()
            self.logger.info('started the \'discover_worker\' thread')
            t2 = threading.Thread(target=self.job_runner, name='job_runner')
            t2.start()
            self.logger.info('started the \'job_runner\' thread')

            self.pyro_daemon = Pyro4.core.Daemon(host=self.host)

            with Pyro4.locateNS(host=self.nameserver, port=self.nameserver_port) as ns:
                uri = self.pyro_daemon.register(self, self.pyro_id)
                ns.register(self.pyro_id, uri)

            self.logger.info('Pyro daemon running on {}'.format(self.pyro_daemon.locationStr))

        self.pyro_daemon.requestLoop()

        with self.discover_cond:
            self.shutdown_all_threads = True
            self.logger.info('shutting down')

            self.runner_cond.notify_all()
            self.discover_cond.notify_all()

            with Pyro4.locateNS(self.nameserver, port=self.nameserver_port) as ns:
                ns.remove(self.pyro_id)

        t1.join()
        self.logger.debug('\'discover_worker\' thread exited')
        t2.join()
        self.logger.debug('\'job_runner\' thread exited')
        self.logger.info('shut down complete')

    def shutdown_all_workers(self, rediscover: bool = False) -> None:
        with self.discover_cond:
            for worker in self.worker_pool.values():
                worker.shutdown()
            if rediscover:
                time.sleep(1)
                self.discover_cond.notify()

    def shutdown(self, shutdown_workers: bool = False) -> None:
        if shutdown_workers:
            self.shutdown_all_workers()

        with self.runner_cond:
            self.pyro_daemon.shutdown()

    @Pyro4.expose
    @Pyro4.oneway
    def trigger_discover_worker(self) -> None:
        # time.sleep(1)
        self.logger.info('A new worker triggered \'discover_worker\'')
        with self.discover_cond:
            self.discover_cond.notify()

    def discover_workers(self) -> None:
        self.discover_cond.acquire()

        while True:
            self.logger.debug('Starting worker discovery')
            update = False

            with Pyro4.locateNS(host=self.nameserver, port=self.nameserver_port) as ns:
                worker_names = ns.list(prefix='{}.worker.'.format(self.run_id))
                self.logger.debug('Found {} potential workers, {} currently in the pool.'.format(
                    len(worker_names), len(self.worker_pool)))

                for wn, uri in worker_names.items():
                    if wn not in self.worker_pool:
                        w = Worker(wn, uri)
                        if not w.is_alive():
                            self.logger.debug('skipping dead worker, {}'.format(wn))
                            continue
                        update = True
                        self.logger.info('discovered new worker, {}'.format(wn))
                        self.worker_pool[wn] = w

            # check the current list of workers
            crashed_jobs: Set[Job] = set()

            all_workers = list(self.worker_pool.keys())
            for wn in all_workers:
                # remove dead entries from the nameserver
                if not self.worker_pool[wn].is_alive():
                    self.logger.info('removing dead worker, {}'.format(wn))
                    update = True
                    # todo check if there were jobs running on that that need to be rescheduled

                    current_job = self.worker_pool[wn].runs_job

                    if current_job is not None:
                        self.logger.info('Job {} was not completed'.format(current_job))
                        crashed_jobs.add(current_job)

                    del self.worker_pool[wn]
                    self.idle_workers.discard(wn)
                    continue

                if not self.worker_pool[wn].is_busy():
                    self.idle_workers.add(wn)

            # try to submit more jobs if something changed
            if update:
                if self.queue_callback is not None:
                    self.discover_cond.release()
                    self.queue_callback(len(self.worker_pool))
                    self.discover_cond.acquire()
                self.runner_cond.notify()

            for crashed_job in crashed_jobs:
                self.discover_cond.release()
                self.register_result(crashed_job.id, Result.failure('Worker died unexpectedly.'))
                self.discover_cond.acquire()

            self.logger.debug('Finished worker discovery')
            self.discover_cond.wait(self.ping_interval)

            if self.shutdown_all_threads:
                self.logger.debug('\'discover_worker\' thread shutting down')
                self.runner_cond.notify()
                self.discover_cond.release()
                return

    def number_of_workers(self) -> int:
        with self.discover_cond:
            return len(self.worker_pool)

    def job_runner(self) -> None:
        self.runner_cond.acquire()
        while True:

            while self.waiting_jobs.empty() or len(self.idle_workers) == 0:
                self.logger.debug('jobs to submit = {}, number of idle workers = {} -> waiting!'.format(
                    self.waiting_jobs.qsize(), len(self.idle_workers)))
                self.runner_cond.wait()
                if self.shutdown_all_threads:
                    self.logger.info('\'job_runner\' thread shutting down')
                    self.discover_cond.notify()
                    self.runner_cond.release()
                    return

            job = self.waiting_jobs.get()
            wn = self.idle_workers.pop()

            worker = self.worker_pool[wn]
            self.logger.debug('starting job {} on {}'.format(str(job.id), worker.name))

            job.time_it('started')
            worker.runs_job = job.id

            worker.proxy.start_computation(self, job.id, config=job.config, info=job.info, budget=job.budget, **job.kwargs)

            job.worker_name = wn
            self.running_jobs[job.id] = job

    def submit_job(self, id: ConfigId,
                   config: Configuration,
                   config_info: ConfigInfo,
                   budget: float,
                   timeout: float,
                   **kwargs
                   ) -> None:
        with self.runner_cond:
            job = Job(id, config, config_info, budget, timeout, **kwargs)
            job.time_it('submitted')
            self.waiting_jobs.put(job)
            self.runner_cond.notify()

    @Pyro4.expose
    @Pyro4.callback
    @Pyro4.oneway
    def register_result(self, id: ConfigId = None, result: Result = None) -> None:
        with self.runner_cond:
            # fill in missing information
            job = self.running_jobs[id]
            job.time_it('finished')
            job.result = result
            job.exception = result.exception

            self.logger.debug('job {} on {} finished'.format(job.id, job.worker_name))

            # delete job
            del self.running_jobs[id]

            # label worker as idle again
            try:
                self.worker_pool[job.worker_name].runs_job = None
                # noinspection PyProtectedMember
                self.worker_pool[job.worker_name].proxy._pyroRelease()
                self.idle_workers.add(job.worker_name)
                # notify the job_runner to check for more jobs to run
                self.runner_cond.notify()
            except KeyError:
                # happens for crashed workers, but we can just continue
                pass

        # call users callback function to register the result
        # needs to be with the condition released, as the master can call
        # submit_job quickly enough to cause a dead-lock
        self.new_result_callback(job)