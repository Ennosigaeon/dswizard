from __future__ import annotations

import time
from enum import Enum
from typing import Optional, Dict, List, TYPE_CHECKING

import numpy as np
from ConfigSpace import ConfigurationSpace
from ConfigSpace.configuration_space import Configuration
from ConfigSpace.read_and_write import json as config_json

from dswizard.core.distance import KdeDistribution, Distance

if TYPE_CHECKING:
    from dswizard.components.pipeline import FlexiblePipeline


class StatusType(Enum):
    """Class to define numbers for status types"""
    SUCCESS = 1
    TIMEOUT = 2
    CRASHED = 3
    ABORT = 4
    MEMOUT = 5
    CAPPED = 6


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
                 runtime: Optional[float] = None,
                 partial_configs: Optional[List[PartialConfig]] = None):
        self.status = status
        self.config = config
        self.loss = loss
        self.runtime = runtime
        if partial_configs is None:
            partial_configs = []

        self.partial_configs: List[PartialConfig] = partial_configs

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
        # TODO dataset properties missing
        # TODO circular imports with FlexiblePipeline
        # FlexiblePipeline.from_list(raw['pipeline'])
        # noinspection PyTypeChecker
        cs = CandidateStructure(config_json.read(raw['configspace']), None,
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


class MetaFeatures:

    def __init__(self, X: np.ndarray):
        self.kde_dist = [KdeDistribution(X[:, i]) for i in range(X.shape[1])]

    def similar(self, other: 'MetaFeatures', epsilon: float = 0.2) -> bool:
        return True
        # TODO distance calculation is too slow

        # distance, phi = Distance.compute_dist(self.kde_dist, other.kde_dist)
        # return distance <= epsilon


class PartialConfig:

    def __init__(self, meta: MetaFeatures, configuration: Configuration, estimator: str):
        self.meta = meta
        self.configuration: Configuration = configuration
        self.estimator = estimator

    def as_dict(self):
        # meta data are serialized via pickle
        # noinspection PyUnresolvedReferences
        return {
            'config': self.configuration.get_dictionary(),
            'configspace': config_json.write(self.configuration.configuration_space),
            # 'meta': pickle.dumps(self.meta),
            'meta': None,
            'estimator': self.estimator,
        }

    @staticmethod
    def from_dict(raw: dict) -> 'PartialConfig':
        # meta data are deserialized via pickle
        config = Configuration(config_json.read(raw['configspace']), raw['config'])
        # noinspection PyTypeChecker
        return PartialConfig(None, config, raw['estimator'])

    def __eq__(self, other):
        if isinstance(other, PartialConfig):
            return self.estimator == other.estimator
        else:
            return self.estimator == other

    def __hash__(self):
        return hash(self.estimator)
