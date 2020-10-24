from __future__ import annotations

import logging
import multiprocessing
import threading
import time
import timeit
from typing import Dict, List, TYPE_CHECKING, Union, Callable

from dswizard.core.model import EvaluationJob, CandidateId, CandidateStructure

if TYPE_CHECKING:
    from dswizard.core.base_structure_generator import BaseStructureGenerator
    from dswizard.core.model import Job, StructureJob
    from dswizard.core.worker import Worker


class NoDaemonProcess(multiprocessing.Process):
    # make 'daemon' attribute always return False
    def _get_daemon(self):
        return False

    def _set_daemon(self, value):
        pass

    daemon = property(_get_daemon, _set_daemon)


class MyPool(multiprocessing.pool.Pool):
    @staticmethod
    def Process(ctx, *args, **kwds):
        return NoDaemonProcess(*args, **kwds)


class Dispatcher:

    def __init__(self,
                 workers: List[Worker],
                 structure_generator: BaseStructureGenerator,
                 logger: logging.Logger = None,
                 ):
        self.structure_generator = structure_generator
        self.shutdown_all_threads = False

        if logger is None:
            self.logger = logging.getLogger('Dispatcher')
        else:
            self.logger = logger

        self.worker_pool: List[Worker] = workers
        self.running_jobs: Dict[CandidateId, Callable] = {}

        self.pool = MyPool(len(self.worker_pool))
        self.condition = threading.Condition()

    def submit_job(self, job: Job, callback: Callable) -> None:
        with self.condition:
            n_running = len(self.running_jobs)
            job.time_submitted = time.time()
            self.running_jobs[job.cid] = callback
            self.pool.apply_async(self._process_job, args=(self.worker_pool[n_running], job),
                                  callback=self._job_callback)
            # Sleep 1 second to ensure process start. Maybe not necessary
            time.sleep(1)

            if len(self.running_jobs) == len(self.worker_pool):
                self.logger.debug('waiting for next worker to be available')
                # TODO infinite waiting is possible
                self.condition.wait()

    def _process_job(self, worker: Worker, job: Union[EvaluationJob, StructureJob]) -> \
            Union[EvaluationJob, CandidateStructure]:
        self.logger.debug('Processing job {}'.format(job.cid))
        job.time_started = time.time()
        worker.runs_job = job.cid

        eval_job = isinstance(job, EvaluationJob)
        if eval_job:
            try:
                result = worker.start_computation(job)
                job.result = result
                # necessary if config was generated on the fly
                job.config = result.config
                self.logger.debug('job {} finished with: {} -> {}'.format(job.cid, result.status, result.loss))
                job.time_finished = timeit.default_timer()
                return job
            except Exception as ex:
                # Catch all. Should never happen
                self.logger.exception('Unhandled exception during job processing: {}'.format(ex))
                return job
        else:
            try:
                cs = self.structure_generator.fill_candidate(job.cs, job.ds, cutoff=job.cutoff, worker=worker)
                job.time_finished = timeit.default_timer()
                self.logger.debug('job {} finished'.format(job.cid))
                return cs
            except Exception as ex:
                # Catch all. Should never happen
                self.logger.exception('Unhandled exception during job processing: {}'.format(ex))
                return job.cs

    def _job_callback(self, result: Union[EvaluationJob, CandidateStructure]):
        try:
            with self.condition:
                callback = self.running_jobs.pop(result.cid)
                callback(result)
                self.condition.notify()
        except Exception as ex:
            self.logger.exception('Unhandled exception in callback: {}'.format(ex))

    def finish_work(self, timeout: float):
        total = len(self.worker_pool)
        deadline = timeit.default_timer() + timeout
        while True:
            now = timeit.default_timer()
            if now > deadline:
                self.logger.warning('Workers did not finish within deadline. {} / {} busy...'.format(busy, total))
                break

            with self.condition:
                busy = len(self.running_jobs)
                self.logger.debug('Waiting for all workers to finish current work. {} / {} busy...'.format(busy, total))
                if busy == 0:
                    break
                else:
                    self.condition.wait(deadline - now)

    def shutdown(self):
        self.pool.close()
        self.pool.join()

    def __getstate__(self):
        # Copy the object's state from self.__dict__ which contains
        # all our instance attributes. Always use the dict.copy()
        # method to avoid modifying the original state.
        state = self.__dict__.copy()
        # Remove the unpicklable entries.
        del state['running_jobs']
        del state['condition']
        del state['pool']
        return state
