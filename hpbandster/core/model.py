import time
from typing import Union, Optional

from ConfigSpace.configuration_space import Configuration, OrderedDict


class ConfigId(object):
    """
    a triplet of ints that uniquely identifies a configuration. the convention is id = (iteration, budget index,
    running index)
    """

    def __init__(self, iteration: int, budget: int, idx: int):
        """
        :param iteration:the iteration of the optimization algorithms. E.g, for Hyperband that is one round of
            Successive Halving
        :param budget: the budget (of the current iteration) for which this configuration was sampled by the
            optimizer. This is only nonzero if the majority of the runs fail and Hyperband resamples to fill empty
            slots, or you use a more 'advanced' optimizer.
        :param idx: this is simply an int >= 0 that sort the configs into the order they where sampled, i.e. (x,x,0) was
            sampled before (x,x,1).
        """

        self.iteration = iteration
        self.budget = budget
        self.idx = idx

    def as_tuple(self):
        return self.iteration, self.budget, self.idx

    def __repr__(self):
        return str(self)

    def __str__(self):
        return str(self.as_tuple())

    def __hash__(self):
        return hash(self.as_tuple())

    def __eq__(self, other):
        return self.as_tuple() == other.as_tuple()

    def __lt__(self, other):
        return self.as_tuple() < other.as_tuple()


class ConfigInfo(object):
    def __init__(self, model_based_pick: bool = False, structure: OrderedDict = None):
        self.model_based_pick = model_based_pick

        if structure is None:
            structure = OrderedDict()
            structure['dummy'] = 'dummy'
        self.structure = structure

    def __repr__(self):
        return str({'model_based_pick': self.model_based_pick,
                    'structure': self.structure})


class Result(object):

    def __init__(self, result: Optional[dict], exception: Optional[str]):
        self.result = result
        self.exception = exception

    @staticmethod
    def success(result: dict):
        return Result(result, None)

    @staticmethod
    def failure(exception: str):
        return Result(None, exception)


class Datum(object):
    def __init__(self,
                 config: Union[dict, Configuration],
                 config_info: dict,
                 results: dict = None,
                 time_stamps: dict = None,
                 exceptions: dict = None,
                 status: str = 'QUEUED',
                 budget: float = 0):

        if isinstance(config, Configuration):
            self.config = config.get_dictionary()
        else:
            self.config = config
        self.config_info = config_info
        self.results = results if results is not None else {}
        self.time_stamps = time_stamps if time_stamps is not None else {}
        self.exceptions = exceptions if exceptions is not None else {}
        self.status = status
        self.budget = budget

    def __repr__(self):
        return str({'config': self.config,
                    'config_info': self.config_info,
                    'losses': '\t'.join(["{}: {}\t".format(k, v['loss']) for k, v in self.results.items()]),
                    'time stamps': self.time_stamps})


class Job(object):
    def __init__(self, id: ConfigId, **kwargs):
        self.id: ConfigId = id

        self.kwargs = kwargs

        self.timestamps = {}

        self.result = None
        self.exception = None

        self.worker_name = None

    def time_it(self, which_time: str) -> None:
        self.timestamps[which_time] = time.time()

    def __repr__(self):
        return str({'job_id': self.id,
                    'kwargs': self.kwargs,
                    'result': self.result,
                    'exception': self.exception})
