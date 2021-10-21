from __future__ import annotations

import os
from collections import namedtuple
from enum import Enum
from typing import Optional, List, TYPE_CHECKING, Tuple, Union

import numpy as np
import openml
from ConfigSpace import ConfigurationSpace
from ConfigSpace.configuration_space import Configuration
from ConfigSpace.read_and_write import json as config_json
from openml import OpenMLClassificationTask
from sklearn.base import BaseEstimator

from dswizard.components.base import EstimatorComponent
from dswizard.components.meta_features import MetaFeatureFactory
from dswizard.util import util

if TYPE_CHECKING:
    from dswizard.pipeline.pipeline import FlexiblePipeline
    from dswizard.components.meta_features import MetaFeatures


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


# Namedtuple instead of class to allow sharing between processes
ConfigKey = namedtuple('ConfigKey', 'hash idx')


class MetaInformation:

    def __init__(self,
                 start_time: float,
                 metric: str,
                 openml_task: int,
                 openml_fold: int,
                 config: dict[str, any]
                 ):
        # Information available before optimization
        self.start_time = start_time
        self.metric = metric
        self.metric_sign = util.metric_sign(self.metric)
        self.openml_task = openml_task
        self.openml_fold = openml_fold
        self.config = config

        # Information available after optimization
        self.end_time: Optional[float] = None
        self.n_structures: Optional[int] = None
        self.n_configs: Optional[int] = None
        self.iterations: Optional[dict] = None
        self.incumbent: Optional[float] = None

    def as_dict(self):
        return {
            'start_time': self.start_time,
            'metric': self.metric,
            'metric_sign': self.metric_sign,
            'openml_task': self.openml_task,
            'openml_fold': self.openml_fold,
            'end_time': self.end_time,
            'n_structures': self.n_structures,
            'n_configs': self.n_configs,
            'iterations': self.iterations,
            'incumbent': self.incumbent,
            'config': self.config
        }


class CandidateId:
    """
    a triplet of ints that uniquely identifies a configuration. the convention is id = (iteration, budget index,
    running index)
    """

    def __init__(self, iteration: int, structure: int, config: int = None):
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

    @property
    def external_name(self):
        if self.config is None:
            return f'{self.iteration:02d}:{self.structure:02d}'
        else:
            return f'{self.iteration:02d}:{self.structure:02d}:{self.config:02d}'

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

    @staticmethod
    def parse(cid: str) -> CandidateId:
        tokens = list(map(lambda x: int(x), cid.split(':')))
        return CandidateId(*tokens)


class Runtime:

    def __init__(self, total: float, timestamp: float):
        self.total = total
        self.timestamp = timestamp

    def as_dict(self):
        return {
            'total': self.total,
            'timestamp': self.timestamp,
        }

    @staticmethod
    def from_dict(raw: dict) -> 'Optional[Runtime]':
        if raw is None:
            return None
        return Runtime(**raw)


class Result:

    def __init__(self,
                 cid: CandidateId,
                 status: Optional[StatusType] = None,
                 config: Configuration = None,
                 loss: Optional[List[float]] = None,
                 runtime: Runtime = None,
                 partial_configs: Optional[List[PartialConfig]] = None,
                 transformed_X: np.ndarray = None):
        self.cid = cid
        self.status = status
        self.config = config

        # structure_loss can be used if a dedicated loss for structure search is necessary
        if loss is None:
            loss = [None]
        self.loss = loss[0]
        self.structure_loss = loss[-1]

        self.runtime = runtime
        self.transformed_X = transformed_X

        if partial_configs is None:
            partial_configs = []
        self.partial_configs: List[PartialConfig] = partial_configs

    def as_dict(self, budget: float = None):
        d = {
            'id': self.cid.external_name,
            'status': self.status.name,
            'loss': [self.loss, self.structure_loss],
            'runtime': self.runtime.as_dict() if self.runtime is not None else None,
            'config': self.config.get_dictionary(),
        }
        if budget is not None:
            d['budget'] = budget
        return d

    @staticmethod
    def from_dict(raw: dict, cs: ConfigurationSpace) -> 'Result':
        return Result(CandidateId.parse(raw['id']), StatusType[raw['status']], Configuration(cs, raw['config']),
                      raw['loss'], Runtime.from_dict(raw['runtime']))


class CandidateStructure:

    def __init__(self,
                 configspace: ConfigurationSpace,
                 pipeline: FlexiblePipeline,
                 cfg_keys: List[ConfigKey],
                 budget: float = 1):
        self.configspace = configspace
        self.pipeline = pipeline
        self.cfg_keys = cfg_keys
        self.budget = budget

        # noinspection PyTypeChecker
        self.cid: CandidateId = None
        self.status: str = 'QUEUED'

        self.results: List[Result] = []

    def get_incumbent(self) -> Optional[Result]:
        if len(self.results) == 0:
            return None
        return min(self.results, key=lambda res: res.loss)

    def add_result(self, result: Result):
        self.results.append(result)

    def __hash__(self):
        return hash(self.configspace)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, CandidateStructure):
            return self.configspace == other.configspace
        return False

    @property
    def steps(self):
        return self.pipeline.steps

    def as_dict(self):
        return {
            'cid': self.cid.without_config().external_name,
            'pipeline': self.pipeline.as_list(),
            'cfg_keys': [(key.hash, key.idx) for key in self.cfg_keys],
            'budget': self.budget,
            'configspace': config_json.write(self.configspace),
        }

    def is_proxy(self):
        return self.configspace is None and self.pipeline is None and self.cfg_keys is None

    @staticmethod
    def from_dict(raw: dict) -> 'CandidateStructure':
        # local import due to circular imports
        from dswizard.pipeline.pipeline import FlexiblePipeline

        # noinspection PyTypeChecker
        cs = CandidateStructure(config_json.read(raw['configspace']), None, raw['cfg_keys'], raw['budget'])
        cs.cid = CandidateId.parse(raw['cid'])
        cs.pipeline = FlexiblePipeline.from_list(raw['pipeline'])
        cs.cfg_keys = [ConfigKey(*t) for t in raw['cfg_keys']]
        return cs

    @staticmethod
    def proxy() -> 'CandidateStructure':
        # noinspection PyTypeChecker
        return CandidateStructure(None, None, None)


class Job:
    # noinspection PyTypeChecker
    def __init__(self, cid: CandidateId, cutoff: float = None):
        self.cid = cid
        self.time_submitted: float = None
        self.time_started: float = None
        self.time_finished: float = None
        self.result: Result = None
        self.cutoff = cutoff


class EvaluationJob(Job):

    def __init__(self,
                 ds: Dataset,
                 candidate_id: CandidateId,
                 cs: Union[CandidateStructure, EstimatorComponent],
                 cutoff: float = None,
                 config: Optional[Configuration] = None,
                 cfg_keys: Optional[List[ConfigKey]] = None):
        super().__init__(candidate_id, cutoff)
        self.ds: Dataset = ds
        self.cs: Union[CandidateStructure, EstimatorComponent] = cs
        self.config = config
        self.cfg_keys = cfg_keys

    # Decorator pattern only used for better readability
    @property
    def component(self) -> Union[BaseEstimator, FlexiblePipeline]:
        if isinstance(self.cs, CandidateStructure):
            return self.cs.pipeline
        else:
            return self.cs


class StructureJob(Job):

    def __init__(self, ds: Dataset, cs: CandidateStructure, cutoff: float = None):
        super().__init__(cs.cid.without_config(), cutoff)
        self.ds = ds
        self.cs = cs


class Dataset:

    def __init__(self,
                 X: np.ndarray,
                 y: np.ndarray,
                 metric: str = 'f1',
                 cutoff: int = 120,
                 task: int = None,
                 fold: int = None,
                 feature_labels: list[str] = None):
        self.X = X
        self.y = y

        if metric not in util.valid_metrics:
            raise KeyError(f'Unknown metric {metric}')
        self.metric = metric
        self.cutoff = cutoff

        self.mf_dict, self.meta_features = MetaFeatureFactory.calculate(X, y, timeout=self.cutoff)

        self.task = task
        self.fold = fold

        self.feature_labels = feature_labels

    @staticmethod
    def from_openml(task: int, fold: int, metric: str):
        # noinspection PyTypeChecker
        task: OpenMLClassificationTask = openml.tasks.get_task(task)
        train_indices, test_indices = task.get_train_test_split_indices(fold=fold)

        X, y = task.get_X_and_y()
        X_train = X[train_indices, :]
        y_train = y[train_indices]
        X_test = X[test_indices, :]
        y_test = y[test_indices]

        feature_labels = list(map(lambda x: x.name, task.get_dataset().features.values()))[:-1]
        ds = Dataset(X_train, y_train, metric=metric, task=task.task_id, fold=fold, feature_labels=feature_labels,)
        ds_test = Dataset(X_test, y_test, metric=metric, task=task.task_id, fold=fold, feature_labels=feature_labels,)
        return ds, ds_test


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

    def __eq__(self, other):
        if isinstance(other, PartialConfig):
            return self.name == other.name
        else:
            return self.name == other

    def __hash__(self):
        return hash(self.name)
