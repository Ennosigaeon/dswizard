import time
from typing import Optional, Dict, Union, List

from ConfigSpace import ConfigurationSpace
from ConfigSpace.configuration_space import Configuration, OrderedDict
from ConfigSpace.read_and_write import json as config_json
from smac.tae.execute_ta_run import StatusType

from dswizard.components.base import ComponentChoice, EstimatorComponent

Structure = Dict[str, Union[ComponentChoice, EstimatorComponent]]


# TODO add fourth index for Configuration number
class CandidateId:
    """
    a triplet of ints that uniquely identifies a configuration. the convention is id = (iteration, budget index,
    running index)
    """

    def __init__(self, iteration: int, structure: int, config: int = -1):
        """
        :param iteration:the iteration of the optimization algorithms. E.g, for Hyperband that is one round of
            Successive Halving
        :param structure: this is simply an int >= 0 that sort the configs into the order they where sampled, i.e. (x,x,0) was
            sampled before (x,x,1).
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


class ConfigInfo:
    def __init__(self, model_based_pick: bool = False, structure: Structure = None):
        self.model_based_pick = model_based_pick

        if structure is None:
            structure = OrderedDict()
        self.structure = structure

    def __repr__(self):
        return str({
            'model_based_pick': self.model_based_pick,
            'structure': self.structure
        })

    def get_dictionary(self) -> dict:
        # TODO serialization of structure is missing
        return {
            'model_based_pick': self.model_based_pick
        }

    @staticmethod
    def from_dictionary(values: dict):
        # TODO deserialization of structure is missing
        return ConfigInfo(model_based_pick=values.get('model_based_pick'))


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
                 structure: Structure,
                 budget: float,
                 timeout: float = None,
                 model_based_pick: bool = False):
        self.configspace = configspace
        self.structure = structure
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
        # TODO structure missing
        return {
            'configspace': config_json.write(self.configspace),
            'structure': None,
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
        # TODO structure missing
        cs = CandidateStructure(config_json.read(raw['configspace']), None, raw['budget'], raw['timeout'],
                                raw['model_based_pick'])
        cs.id = CandidateId(*raw['id'])
        cs.status = raw['status']
        # cs.results = {k: [Result.from_dict(res) for res in v] for k, v in raw['results'].items()},
        cs.timestamps = raw['timestamps']
        return cs


class Job:
    # noinspection PyTypeChecker
    def __init__(self,
                 candidate_id: CandidateId,
                 config: Configuration,
                 configspace: ConfigurationSpace,
                 structure: Structure,
                 budget: float,
                 timout: float,
                 **kwargs):
        self.id = candidate_id
        self.config = config

        # TODO configspace only added as a temporary workaround. Structure should be sufficient
        self.configspace = configspace
        self.structure = structure
        self.budget = budget
        self.timeout = timout
        self.kwargs = kwargs

        self.worker_name: str = None
        self.time_submitted: float = None
        self.time_started: float = None
        self.time_finished: float = None
        self.result: Result = None
