import time

from ConfigSpace.configuration_space import ConfigurationSpace


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
        if not isinstance(other, ConfigId):
            return False
        return self.as_tuple() == other.as_tuple()


class Datum(object):
    def __init__(self,
                 config: ConfigurationSpace,
                 config_info: dict,
                 results: dict = None,
                 time_stamps: dict = None,
                 exceptions: dict = None,
                 status: str = 'QUEUED',
                 budget: float = 0):
        self.config = config.get_dictionary()
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
        return "job_id: {}\n" \
               "kwargs: {}\n" \
               "result: {}\n" \
               "exception: {}\n".format(self.id, self.kwargs, self.result, self.exception)
