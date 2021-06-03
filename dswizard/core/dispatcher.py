from __future__ import annotations

import logging
import multiprocessing
import threading
import time
import timeit
from typing import Dict, List, TYPE_CHECKING, Union, Callable

from dswizard.core.model import EvaluationJob, StructureJob, CandidateId, CandidateStructure

if TYPE_CHECKING:
    from dswizard.core.base_structure_generator import BaseStructureGenerator
    from dswizard.core.model import Job
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
    def Process(ctx, *args, **kwargs):
        return NoDaemonProcess(*args, **kwargs)


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

        if len(self.worker_pool) > 1:
            self.pool = MyPool(len(self.worker_pool))
        else:
            self.pool = None
        self.condition = threading.Condition()

    def submit_job(self, job: Job, callback: Callable) -> None:
        with self.condition:
            n_running = len(self.running_jobs)
            job.time_submitted = time.time()
            self.running_jobs[job.cid] = callback

            if len(self.worker_pool) > 1:
                self.pool.apply_async(self._process_job, args=(self.worker_pool[n_running], job),
                                      callback=self._job_callback)
                # Sleep 1 second to ensure process start. Maybe not necessary
                time.sleep(1)
            else:
                res = self._process_job(self.worker_pool[0], job)
                self._job_callback(res)

            if len(self.running_jobs) == len(self.worker_pool):
                self.logger.debug('waiting for next worker to be available')
                # TODO infinite waiting is possible
                self.condition.wait()

    def _process_job(self, worker: Worker, job: Job) -> \
            Union[EvaluationJob, CandidateStructure]:
        self.logger.debug(f'Processing job {job.cid}')
        job.time_started = time.time()
        worker.runs_job = job.cid

        if isinstance(job, EvaluationJob):
            try:
                result = worker.start_computation(job)
                job.result = result
                # necessary if config was generated on the fly
                job.config = result.config
                self.logger.debug(f'job {job.cid} finished with: {result.status} -> {result.loss}')
                job.time_finished = timeit.default_timer()
                return job
            except Exception as ex:
                # Catch all. Should never happen
                self.logger.exception(f'Unhandled exception during job processing: {ex}')
                return job
        elif isinstance(job, StructureJob):
            try:
                cs = self.structure_generator.fill_candidate(job.cs, job.ds, cutoff=job.cutoff, worker=worker)
                job.time_finished = timeit.default_timer()
                self.logger.debug(f'job {job.cid} finished')
                return cs
            except Exception as ex:
                # Catch all. Should never happen
                self.logger.exception(f'Unhandled exception during job processing: {ex}')
                return job.cs
        else:
            raise ValueError(f'Unknown Job type {job}')

    def _job_callback(self, result: Union[EvaluationJob, CandidateStructure]):
        try:
            with self.condition:
                callback = self.running_jobs.pop(result.cid)
                callback(result)
                self.condition.notify()
        except Exception as ex:
            self.logger.exception(f'Unhandled exception in callback: {ex}')

    def finish_work(self, timeout: float):
        total = len(self.worker_pool)
        deadline = timeit.default_timer() + timeout
        while True:
            now = timeit.default_timer()
            with self.condition:
                busy = len(self.running_jobs)
                self.logger.debug(f'Waiting for all workers to finish current work. {busy} / {total} busy...')
                if busy == 0:
                    break
                else:
                    self.condition.wait(deadline - now)
            if now > deadline:
                self.logger.warning(f'Workers did not finish within deadline. {busy} / {total} busy...')
                break

    def shutdown(self):
        if self.pool is not None:
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
