from __future__ import annotations

import logging
import queue
import threading
import time
from typing import Dict, Set, List, TYPE_CHECKING

if TYPE_CHECKING:
    from dswizard.core.model import Job
    from dswizard.core.worker import Worker


class Dispatcher:

    def __init__(self,
                 workers: List[Worker],
                 logger: logging.Logger = None,
                 ):
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

            self.logger.debug('starting job {} on {}'.format(str(job.cid), worker.worker_id))
            t = threading.Thread(target=self._compute_result, args=(worker, job))
            t.start()

    def shutdown(self) -> None:
        self.shutdown_all_threads = True
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

    def _compute_result(self, worker: Worker, job: Job) -> None:
        job.time_started = time.time()
        worker.runs_job = job.cid

        result = worker.start_computation(job)

        with self.runner_cond:
            # fill in missing information
            job.time_finished = time.time()
            job.result = result
            # necessary if config was generated on the fly
            job.config = result.config

            self.logger.debug('job {} on {} finished'.format(job.cid, worker.worker_id))

            # label worker as idle again
            try:
                self.idle_workers.add(worker.worker_id)
                # notify the job_runner to check for more jobs to run
                self.runner_cond.notify()
            except KeyError:
                # happens for crashed workers, but we can just continue
                pass

        # call users callback function to register the result
        # needs to be with the condition released, as the master can call
        # submit_job quickly enough to cause a dead-lock
        job.callback(job)

        with self.callback_cond:
            self.callback_cond.notify()

    def finish_work(self):
        # TODO remove busy waiting
        while len(self.idle_workers) < len(self.worker_pool):
            self.logger.debug('{} worker busy...'.format(len(self.worker_pool) - len(self.idle_workers)))
            time.sleep(10)
