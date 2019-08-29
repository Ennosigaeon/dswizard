import time
from typing import Optional, Dict, List

from ConfigSpace import ConfigurationSpace
from ConfigSpace.configuration_space import Configuration
from ConfigSpace.read_and_write import json as config_json
from smac.tae.execute_ta_run import StatusType

from dswizard.components.pipeline import FlexiblePipeline


class CandidateId:
    """
    a triplet of ints that uniquely identifies a configuration. the convention is id = (iteration, budget index,
    running index)
    """

    def __init__(self, iteration: int, structure: int, config: int = -1):
        """
        :param iteration:the iteration of the optimization algorithms. E.g, for Hyperband that is one round of
            Successive Halving
        :param structure: this is simply an int >= 0 that sort the configs into the order they where sampled, i.e.
            (x,x,0) was sampled before (x,x,1).
        :param config: the budget (of the current iteration) for which this configuration was sampled by the
            optimizer. This is only nonzero if the majority of the runs fail and Hyperband resamples to fill empty
            slots, or you use a more 'advanced' optimizer.
        """

        self.iteration = iteration
        self.structure = structure
        self.config = config

    def as_tuple(self):
        return self.iteration, self.structure, self.config

    def with_config(self, config: int) -> 'CandidateId':
        return CandidateId(self.iteration, self.structure, config)

    def without_config(self) -> 'CandidateId':
        return CandidateId(self.iteration, self.structure)

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


class Result:

    def __init__(self,
                 status: Optional[StatusType] = None,
                 config: Configuration = None,
                 loss: Optional[float] = None,
                 runtime: Optional[float] = None):
        self.status = status
        self.config = config
        self.loss = loss
        self.runtime = runtime

    def as_dict(self):
        return {
            'status': str(self.status),
            'loss': self.loss,
            'runtime': self.runtime,
            'config': self.config.get_dictionary()
        }

    @staticmethod
    def from_dict(raw: dict) -> 'Result':
        return Result(**raw)


class CandidateStructure:

    def __init__(self,
                 configspace: ConfigurationSpace,
                 pipeline: FlexiblePipeline,
                 budget: float,
                 timeout: int = None,
                 model_based_pick: bool = False):
        self.configspace = configspace
        self.pipeline = pipeline
        self.budget = budget
        self.timeout = timeout
        self.model_based_pick = model_based_pick

        # noinspection PyTypeChecker
        self.id: CandidateId = None
        self.status: str = 'QUEUED'

        self.results: Dict[float, List[Result]] = {}
        self.timestamps: Dict[str, float] = {}

    def time_it(self, which_time: str) -> None:
        self.timestamps[which_time] = time.time()

    def get_incumbent(self, budget: float = None) -> Result:
        if budget is None:
            budget = max(self.results.keys())
        return min(self.results[budget], key=lambda res: res.loss)

    def add_result(self, budget: float, result: Result):
        self.results.setdefault(budget, []).append(result)

    def as_dict(self):
        return {
            'configspace': config_json.write(self.configspace),
            'pipeline': self.pipeline.as_list(),
            'budget': self.budget,
            'timeout': self.timeout,
            'model_based_pick': self.model_based_pick,
            'id': self.id.as_tuple(),
            'status': self.status,
            'results': {k: [res.as_dict() for res in v] for k, v in self.results.items()},
            'timestamps': self.timestamps
        }

    @staticmethod
    def from_dict(raw: dict) -> 'CandidateStructure':
        cs = CandidateStructure(config_json.read(raw['configspace']), FlexiblePipeline.from_list(raw['pipeline']),
                                raw['budget'], raw['timeout'], raw['model_based_pick'])
        cs.id = CandidateId(*raw['id'])
        cs.status = raw['status']
        # cs.results = {k: [Result.from_dict(res) for res in v] for k, v in raw['results'].items()},
        cs.timestamps = raw['timestamps']
        return cs


class Job:
    # noinspection PyTypeChecker
    def __init__(self,
                 candidate_id: CandidateId,
                 config: Optional[Configuration],
                 pipeline: FlexiblePipeline,
                 budget: float,
                 timout: float,
                 **kwargs):
        self.id = candidate_id
        self.config = config

        self.pipeline = pipeline
        self.budget = budget
        self.timeout = timout
        self.kwargs = kwargs

        self.worker_name: str = None
        self.time_submitted: float = None
        self.time_started: float = None
        self.time_finished: float = None
        self.result: Result = None
