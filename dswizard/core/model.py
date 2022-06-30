from __future__ import annotations

import re
from collections import namedtuple
from enum import Enum
from typing import Optional, List, TYPE_CHECKING, Tuple, Union, Any, Dict

import joblib
import numpy as np
from ConfigSpace import ConfigurationSpace
from ConfigSpace.configuration_space import Configuration
from ConfigSpace.read_and_write import json as config_json
from sklearn.base import BaseEstimator

import dswizard.components.util as comp_util
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
                 data_file: str,
                 config: Dict[str, Any]
                 ):
        # Information available before optimization
        self.start_time = start_time
        self.metric = metric
        self.is_minimization = util.metric_sign(self.metric) == 1
        self.openml_task = openml_task
        self.openml_fold = openml_fold
        self.data_file = data_file
        self.config = config

        # Information available after optimization
        self.end_time: Optional[float] = None
        self.n_structures: Optional[int] = None
        self.n_configs: Optional[int] = None
        self.iterations: Optional[Dict] = None
        self.incumbent: Optional[float] = None

    def as_dict(self):
        return {
            'start_time': self.start_time,
            'metric': self.metric,
            'is_minimization': self.is_minimization,
            'openml_task': self.openml_task,
            'openml_fold': self.openml_fold,
            'end_time': self.end_time,
            'n_structures': self.n_structures,
            'n_configs': self.n_configs,
            'iterations': self.iterations,
            'data_file': self.data_file,
            'incumbent': self.incumbent,
            'config': self.config
        }


class CandidateId:
    """
    a triplet of ints that uniquely identifies a configuration. the convention is id = (iteration, budget index,
    running index)
    """

    def __init__(self, iteration: int, structure: int, config: Union[int, str] = None):
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

    def with_config(self, config: Union[int, str]) -> 'CandidateId':
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
        if isinstance(other, CandidateId):
            return self.as_tuple() == other.as_tuple()
        elif isinstance(other, tuple):
            return self.as_tuple() == other
        else:
            return False

    def __lt__(self, other):
        return self.as_tuple() < other.as_tuple()

    @staticmethod
    def parse(cid: str) -> CandidateId:
        tokens = list(map(lambda x: int(x), cid.split(':')))
        return CandidateId(*tokens)

    @staticmethod
    def from_model_file(name: str) -> CandidateId:
        sub_string = re.search(r'(\d+-\d+-\d+)', name).group(1)
        return CandidateId(*map(int, sub_string.split('-')))


class Runtime:

    def __init__(self, training_time: float, timestamp: float):
        self.training_time = training_time
        self.timestamp = timestamp

    def as_dict(self):
        return {
            'training_time': self.training_time,
            'timestamp': self.timestamp,
        }

    @staticmethod
    def from_dict(raw: Dict) -> 'Optional[Runtime]':
        if raw is None:
            return None
        return Runtime(**raw)


class Result:

    def __init__(self,
                 cid: CandidateId,
                 status: Optional[StatusType] = None,
                 config: Configuration = None,
                 loss: Optional[float] = None,
                 structure_loss: Optional[float] = None,
                 runtime: Runtime = None,
                 partial_configs: Optional[List[PartialConfig]] = None,
                 transformed_X: np.ndarray = None):
        self.cid = cid
        self.status = status
        self.config = config

        # structure_loss can be used if a dedicated loss for structure search is necessary
        if structure_loss is None:
            structure_loss = loss
        self.loss = loss
        self.structure_loss = structure_loss

        self.runtime = runtime
        self.transformed_X = transformed_X

        if partial_configs is None:
            partial_configs = []
        self.partial_configs: List[PartialConfig] = partial_configs

        self.model_file: Optional[str] = None

    def as_dict(self, budget: float = None, loss_sign: float = 1):
        d = {
            'model_file': self.model_file,
            'id': self.cid.external_name,
            'status': self.status.name,
            'loss': self.loss * loss_sign,
            'structure_loss': self.structure_loss,  # by definition always a min. problem, no need to adjust sign
            'runtime': self.runtime.as_dict() if self.runtime is not None else None,
            'config': self.config.get_dictionary(),
            'origin': self.config.origin if self.config is not None else None,
        }
        if budget is not None:
            d['budget'] = budget
        return d

    @staticmethod
    def from_dict(raw: Dict, cs: ConfigurationSpace) -> 'Result':
        config = Configuration(cs, raw['config'])
        config.origin = raw['origin']
        return Result(CandidateId.parse(raw['id']), StatusType[raw['status']], config,
                      raw['loss'], raw['structure_loss'], Runtime.from_dict(raw['runtime']))


class CandidateStructure:

    def __init__(self,
                 configspace: Optional[ConfigurationSpace],
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
            'pipeline': comp_util.serialize(self.pipeline),
            'cfg_keys': [(key.hash, key.idx) for key in self.cfg_keys],
            'budget': self.budget,
            'configspace': config_json.write(self.configspace) if self.configspace is not None else None,
        }

    def is_proxy(self):
        return self.configspace is None and self.pipeline is None and self.cfg_keys is None

    @staticmethod
    def from_dict(raw: Dict) -> 'CandidateStructure':
        # noinspection PyTypeChecker
        cs = CandidateStructure(config_json.read(raw['configspace']), None, raw['cfg_keys'], raw['budget'])
        cs.cid = CandidateId.parse(raw['cid'])
        cs.pipeline = comp_util.deserialize(**raw['pipeline'])
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
                 feature_names: List[str] = None):
        self.X = X
        self.y = y

        if metric not in util.valid_metrics:
            raise KeyError(f'Unknown metric {metric}')
        self.metric = metric
        self.cutoff = cutoff

        self.mf_dict, self.meta_features = MetaFeatureFactory.calculate(X, y, timeout=self.cutoff)

        self.task = task
        self.fold = fold

        self.feature_names = feature_names

    def store(self, file_name: str):
        joblib.dump((self.X, self.y, self.feature_names), file_name)

    @staticmethod
    def from_openml(task: int, fold: int, metric: str):
        import openml
        # noinspection PyTypeChecker
        task: openml.tasks.OpenMLClassificationTask = openml.tasks.get_task(task)
        train_indices, test_indices = task.get_train_test_split_indices(fold=fold)

        X, y = task.get_X_and_y(dataset_format='dataframe')
        X_train = X.values[train_indices, :]
        y_train = y.values[train_indices]
        X_test = X.values[test_indices, :]
        y_test = y.values[test_indices]

        feature_names = X.columns.tolist()
        ds = Dataset(X_train, y_train, metric=metric, task=task.task_id, fold=fold, feature_names=feature_names)
        ds_test = Dataset(X_test, y_test, metric=metric, task=task.task_id, fold=fold, feature_names=feature_names)
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
    def from_dict(raw: Dict, origin: str) -> 'PartialConfig':
        # meta data are deserialized via pickle
        config = Configuration(config_json.read(raw['configspace']), vector=np.array(raw['config']))
        config.origin = origin
        # noinspection PyTypeChecker
        return PartialConfig(raw['cfg_key'], config, raw['name'], np.array(raw['mf']))

    def __eq__(self, other):
        if isinstance(other, PartialConfig):
            return self.name == other.name
        else:
            return self.name == other

    def __hash__(self):
        return hash(self.name)
