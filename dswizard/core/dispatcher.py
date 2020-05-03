from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Callable, Dict, Set, List, TYPE_CHECKING

from dswizard.core.model import Result

if TYPE_CHECKING:
    from dswizard.core.model import CandidateId, Job
    from dswizard.core.worker import Worker


class Dispatcher:

    def __init__(self,
                 workers: List[Worker],
                 new_result_callback: Callable[[Job], None],
                 run_id: str = '0',
                 logger: logging.Logger = None,
                 ):
        self.new_result_callback = new_result_callback
        self.run_id = run_id
        self.shutdown_all_threads = False

        if logger is None:
            self.logger = logging.getLogger('Dispatcher')
        else:
            self.logger = logger

        self.worker_pool: Dict[str, Worker] = {}
        self.idle_workers: Set[str] = set()
        for worker in workers:
            self.idle_workers.add(worker.worker_id)
            self.worker_pool[worker.worker_id] = worker

        self.waiting_jobs: queue.Queue[Job] = queue.Queue()
        self.running_jobs: Dict[CandidateId, Job] = {}

        self.runner_cond = threading.Condition()
        self.callback_cond = threading.Condition()
        self.shutdown_cond = threading.Condition()

    def run(self) -> None:
        t1 = threading.Thread(target=self._job_runner, name='job_runner')
        t1.start()
        self.logger.info('started the \'job_runner\' thread')

        with self.shutdown_cond:
            self.shutdown_cond.wait()

        self.shutdown_all_threads = True
        self.logger.info('shutting down')
        with self.runner_cond:
            self.runner_cond.notify_all()

        t1.join()
        self.logger.debug('\'job_runner\' thread exited')
        self.logger.info('shut down complete')

    def _job_runner(self) -> None:
        self.runner_cond.acquire()
        while True:

            while self.waiting_jobs.empty() or len(self.idle_workers) == 0:
                self.logger.debug('jobs to submit = {}, number of idle workers = {} -> waiting!'.format(
                    self.waiting_jobs.qsize(), len(self.idle_workers)))
                self.runner_cond.wait()
                if self.shutdown_all_threads:
                    self.logger.info('\'job_runner\' thread shutting down')
                    self.runner_cond.release()
                    return

            job = self.waiting_jobs.get()
            wn = self.idle_workers.pop()
            worker = self.worker_pool[wn]

            # There are still idle workers available. Immediately allow starting next job
            if len(self.idle_workers) > 0:
                with self.callback_cond:
                    self.callback_cond.notify()

            self.logger.debug('starting job {} on {}'.format(str(job.id), worker.worker_id))

            job.time_started = time.time()
            worker.runs_job = job.id

            job.worker_name = worker.worker_id
            self.running_jobs[job.id] = job

            t = threading.Thread(target=worker.start_computation, args=(self, job))
            t.start()

    def shutdown(self) -> None:
        with self.shutdown_cond:
            self.shutdown_cond.notify()

    def number_of_workers(self) -> int:
        return len(self.worker_pool)

    def submit_job(self, job: Job) -> None:
        with self.runner_cond:
            job.time_submitted = time.time()
            self.waiting_jobs.put(job)
            self.runner_cond.notify()

        with self.callback_cond:
            self.logger.debug('waiting for next worker to be available')
            self.callback_cond.wait()

    def register_result(self, id: CandidateId = None, result: Result = None) -> None:
        with self.runner_cond:
            # fill in missing information
            job = self.running_jobs[id]
            job.time_finished = time.time()
            job.result = result
            # necessary if config was generated on the fly
            job.config = result.config

            self.logger.debug('job {} on {} finished'.format(job.id, job.worker_name))

            # delete job
            del self.running_jobs[id]

            # label worker as idle again
            try:
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

        with self.callback_cond:
            self.callback_cond.notify()
