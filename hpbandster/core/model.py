import time
from typing import Tuple


class Datum(object):
    def __init__(self,
                 config: dict,
                 config_info: dict,
                 results: dict = None,
                 time_stamps: dict = None,
                 exceptions: dict = None,
                 status: str = 'QUEUED',
                 budget: float = 0):
        self.config = config
        self.config_info = config_info
        self.results = results if results is not None else {}
        self.time_stamps = time_stamps if time_stamps is not None else {}
        self.exceptions = exceptions if exceptions is not None else {}
        self.status = status
        self.budget = budget

    def __repr__(self):
        return (
                "\nconfig:{}\n".format(self.config) +
                "config_info:\n{}\n".format(self.config_info) +
                "losses:\n"
                '\t'.join(["{}: {}\t".format(k, v['loss']) for k, v in self.results.items()]) +
                "time stamps: {}".format(self.time_stamps)
        )


class Job(object):
    def __init__(self, id: Tuple[int, int, int], **kwargs):
        self.id: Tuple[int, int, int] = id

        self.kwargs = kwargs

        self.timestamps = {}

        self.result = None
        self.exception = None

        self.worker_name = None

    def time_it(self, which_time: str) -> None:
        self.timestamps[which_time] = time.time()

    def __repr__(self):
        return "job_id: {}\n" \
               "kwargs: {}\n" \
               "result: {}\n" \
               "exception: {}\n".format(self.id, self.kwargs, self.result, self.exception)
