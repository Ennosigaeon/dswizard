from __future__ import annotations

import time
from enum import Enum
from typing import Optional, Dict, List, TYPE_CHECKING, Tuple, Union

import numpy as np
from ConfigSpace import ConfigurationSpace
from ConfigSpace.configuration_space import Configuration
from ConfigSpace.read_and_write import json as config_json
from sklearn.base import BaseEstimator

from automl.components.base import EstimatorComponent
from automl.components.meta_features import MetaFeatureFactory
from dswizard.util.util import prefixed_name

if TYPE_CHECKING:
    from dswizard.components.pipeline import FlexiblePipeline
    from automl.components.meta_features import MetaFeatures


class StatusType(Enum):
    """Class to define numbers for status types"""
    SUCCESS = 1
    TIMEOUT = 2
    CRASHED = 3
    ABORT = 4
    MEMOUT = 5
    CAPPED = 6
    INEFFECTIVE = 7
    DUPLICATE = 8


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


class Runtime:

    def __init__(self, total: float, fit: float = 0, config: float = 0):
        self.total = total
        self.fit = fit
        self.config = config

    def as_dict(self):
        return {
            'total': self.total,
            'fit': self.fit,
            'conf': self.config,
        }

    @staticmethod
    def from_dict(raw: dict) -> 'Runtime':
        return Runtime(**raw)


class Result:

    def __init__(self,
                 status: Optional[StatusType] = None,
                 config: Configuration = None,
                 loss: Optional[float] = None,
                 runtime: Runtime = None,
                 partial_configs: Optional[List[PartialConfig]] = None,
                 transformed_X: np.ndarray = None):
        self.status = status
        self.config = config
        self.loss = loss
        self.runtime = runtime
        self.transformed_X = transformed_X

        if partial_configs is None:
            partial_configs = []
        self.partial_configs: List[PartialConfig] = partial_configs

    def as_dict(self):
        return {
            'status': str(self.status),
            'loss': self.loss,
            'runtime': self.runtime.as_dict(),
            'config': self.config.get_dictionary(),
        }

    @staticmethod
    def from_dict(raw: dict) -> 'Result':
        return Result(raw['status'], raw['config'], raw['loss'], Runtime.from_dict(raw['runtime']))


class CandidateStructure:

    def __init__(self,
                 configspace: ConfigurationSpace,
                 pipeline: FlexiblePipeline,
                 cfg_keys: List[Tuple[float, int]],
                 budget: float = 1,
                 model_based_pick: bool = False):
        self.configspace = configspace
        self.pipeline = pipeline
        self.cfg_keys = cfg_keys
        self.budget = budget
        self.model_based_pick = model_based_pick

        # noinspection PyTypeChecker
        self.cid: CandidateId = None
        self.status: str = 'QUEUED'

        self.results: List[Result] = []
        self.timestamps: Dict[str, float] = {}

    def time_it(self, which_time: str) -> None:
        self.timestamps[which_time] = time.time()

    def get_incumbent(self) -> Optional[Result]:
        if len(self.results) == 0:
            return None
        return min(self.results, key=lambda res: res.loss)

    def add_result(self, result: Result):
        self.results.append(result)

    @property
    def steps(self):
        return self.pipeline.steps

    def as_dict(self):
        return {
            'configspace': config_json.write(self.configspace),
            'pipeline': self.pipeline.as_list(),
            'cfg_keys': self.cfg_keys,
            'budget': self.budget,
            'model_based_pick': self.model_based_pick,
            'cid': self.cid.as_tuple(),
            'status': self.status,
            'results': [res.as_dict() for res in self.results],
            'timestamps': self.timestamps
        }

    @staticmethod
    def from_dict(raw: dict) -> 'CandidateStructure':
        # TODO dataset properties missing
        # TODO circular imports with FlexiblePipeline
        # FlexiblePipeline.from_list(raw['pipeline'])
        # noinspection PyTypeChecker
        cs = CandidateStructure(config_json.read(raw['configspace']), None, raw['cfg_keys'],
                                raw['budget'], raw['model_based_pick'])
        cs.cid = CandidateId(*raw['cid'])
        cs.status = raw['status']
        cs.results = [Result.from_dict(res) for res in raw['results']],
        cs.timestamps = raw['timestamps']
        return cs


class Job:
    # noinspection PyTypeChecker
    def __init__(self,
                 ds: Dataset,
                 candidate_id: CandidateId,
                 cs: Union[CandidateStructure, EstimatorComponent],
                 cutoff: float = None,
                 config: Optional[Configuration] = None,
                 cfg_keys: Optional[List[Tuple[float, int]]] = None,
                 **kwargs):
        self.ds = ds
        self.cid = candidate_id
        self.cs = cs
        self.cutoff = cutoff
        self.config = config
        self.cfg_keys = cfg_keys

        self.kwargs = kwargs

        self.worker_name: str = None
        self.time_submitted: float = None
        self.time_started: float = None
        self.time_finished: float = None
        self.result: Result = None

    # Decorator pattern only used for better readability
    @property
    def component(self) -> Union[FlexiblePipeline, BaseEstimator]:
        if isinstance(self.cs, CandidateStructure):
            return self.cs.pipeline
        else:
            return self.cs


class Dataset:

    def __init__(self,
                 X: np.ndarray,
                 y: np.ndarray):
        self.X = X
        self.y = y

        self.mf_dict, self.meta_features = MetaFeatureFactory.calculate(X, y)


class PartialConfig:

    def __init__(self, cfg_key: Tuple[float, int],
                 configuration: Configuration,
                 name: str,
                 mf: Optional[MetaFeatures]):
        self.cfg_key = cfg_key
        self.config: Configuration = configuration
        self.name = name

        if mf is None:
            mf = np.zeros((1, 1))
        self.mf = mf

    def is_empty(self):
        # noinspection PyUnresolvedReferences
        return len(self.config.configuration_space.get_hyperparameters()) == 0

    def as_dict(self):
        # meta data are serialized via pickle
        # noinspection PyUnresolvedReferences
        return {
            'config': self.config.get_array().tolist(),
            'configspace': config_json.write(self.config.configuration_space),
            'cfg_key': self.cfg_key,
            'name': self.name,
            'mf': self.mf.tolist()
        }

    @staticmethod
    def from_dict(raw: dict) -> 'PartialConfig':
        # meta data are deserialized via pickle
        config = Configuration(config_json.read(raw['configspace']), vector=np.array(raw['config']))
        # noinspection PyTypeChecker
        return PartialConfig(raw['cfg_key'], config, raw['name'], np.array(raw['mf']))

    @staticmethod
    def merge(partial_configs: List[PartialConfig]):
        if len(partial_configs) == 1:
            return partial_configs[0].config

        complete = {}
        cs = ConfigurationSpace()

        for partial_config in partial_configs:
            for param, value in partial_config.config.get_dictionary().items():
                param = prefixed_name(partial_config.name, param)
                complete[param] = value
            cs.add_configuration_space(partial_config.name, partial_config.config.configuration_space)

        return Configuration(cs, complete)

    def __eq__(self, other):
        if isinstance(other, PartialConfig):
            return self.name == other.name
        else:
            return self.name == other

    def __hash__(self):
        return hash(self.name)
