from __future__ import annotations

import timeit
from typing import Dict, List, Tuple, Union, TYPE_CHECKING, Optional

import numpy as np
from ConfigSpace.configuration_space import Configuration, ConfigurationSpace
from sklearn.base import BaseEstimator, clone
from sklearn.pipeline import Pipeline, _fit_transform_one
from sklearn.utils import _print_elapsed_time

from dswizard.components.base import ComponentChoice, EstimatorComponent
from dswizard.core.model import PartialConfig
from dswizard.util import util
from dswizard.util.util import prefixed_name

if TYPE_CHECKING:
    from dswizard.core.logger import ProcessLogger
    from dswizard.core.config_cache import ConfigCache
    from dswizard.components.meta_features import MetaFeatures


class FlexiblePipeline(Pipeline, BaseEstimator):

    def __init__(self,
                 steps: List[Tuple[str, EstimatorComponent]],
                 configuration: Optional[dict] = None,
                 cfg_cache: Optional[ConfigCache] = None,
                 cfg_keys: Optional[List[Tuple[float, int]]] = None):
        self.configuration = None
        self.cfg_keys = cfg_keys
        self.cfg_cache: Optional[ConfigCache] = cfg_cache

        # super.__init__ has to be called after initializing all properties provided in constructor
        super().__init__(steps, verbose=False)
        self.steps_ = dict(steps)
        self.configuration_space: ConfigurationSpace = self.get_hyperparameter_search_space(mf=None)

        self.fit_time = 0
        self.config_time = 0

        if configuration is not None:
            self.set_hyperparameters(configuration)

    def to_networkx(self, prefix: str = None):
        import networkx as nx
        G = nx.DiGraph()
        predecessor = None
        for name, estimator in self.steps_.items():
            name = prefixed_name(prefix, name)

            if isinstance(estimator, SubPipeline):
                split = name
                name = '{}_merge'.format(name)

                G.add_node(split, label='split', name=name)
                G.add_node(name, label='merge')

                for p_name, p in estimator.pipelines.items():
                    prefix = prefixed_name(split, p_name)
                    H = p.to_networkx(prefix=prefix)
                    G = nx.compose(G, H)
                    G.add_edge(split, prefixed_name(prefix, p.steps[0][0]))
                    G.add_edge(prefixed_name(prefix, p.steps[-1][0]), name)
            else:
                G.add_node(name, label=estimator.name().split('.')[-1], name=name)

            if predecessor is not None:
                G.add_edge(predecessor, name)
            predecessor = name
        return G

    def get_step(self, name: str):
        tokens = name.split(':')
        step_name = tokens[0]

        estimator = self.steps_[step_name]
        if isinstance(estimator, SubPipeline) and len(tokens) > 1:
            pipeline_name = tokens[1]

            n_prefix = len(step_name) + 1 + len(pipeline_name) + 1
            return estimator.pipelines[pipeline_name].get_step(name[n_prefix:])
        return estimator

    def all_names(self, prefix: str = None, exclude_parents: bool = False) -> List[str]:
        res = []
        for name, component in self.steps_.items():
            n = prefixed_name(prefix, name)
            if isinstance(component, SubPipeline):
                if not exclude_parents:
                    res.append(n)

                for p_name, p in component.pipelines.items():
                    res.extend(p.all_names(prefixed_name(name, p_name), exclude_parents))
            else:
                res.append(n)
        return res

    def _validate_steps(self):
        if len(self.steps) == 0:
            raise TypeError('Pipeline has to contain at least 1 step')
        super()._validate_steps()
        if not hasattr(self.steps[-1][1], 'predict'):
            raise TypeError('Last step of Pipeline should implement predict.')

    def _fit(self,
             X: np.ndarray,
             y: np.ndarray = None,
             logger: ProcessLogger = None,
             prefix: str = None,
             **fit_params: dict):
        # shallow copy of steps - this should really be steps_
        self.steps = list(self.steps)
        self._validate_steps()

        fit_params_steps = {name: {} for name, step in self.steps
                            if step is not None}
        for pname, pval in fit_params.items():
            if '__' not in pname:
                raise ValueError(
                    "Pipeline.fit does not accept the {} parameter. "
                    "You can pass parameters to specific steps of your "
                    "pipeline using the stepname__parameter format, e.g. "
                    "`Pipeline.fit(X, y, logisticregression__sample_weight"
                    "=sample_weight)`.".format(pname))
            step, param = pname.split('__', 1)
            fit_params_steps[step][param] = pval
        Xt = X
        for (step_idx,
             name,
             transformer) in self._iter(with_final=False,
                                        filter_passthrough=False):
            if transformer is None or transformer == 'passthrough':
                with _print_elapsed_time('Pipeline',
                                         self._log_message(step_idx)):
                    continue

            cloned_transformer = clone(transformer)

            # Configure transformer on the fly if necessary
            if self.configuration is None:
                config: Configuration = self._get_config_for_step(step_idx, prefix, name, logger)
                cloned_transformer.set_hyperparameters(configuration=config.get_dictionary())

            # Fit or load from cache the current transformer
            if isinstance(cloned_transformer, SubPipeline):
                Xt, fitted_transformer = _fit_transform_one(
                    cloned_transformer, Xt, y, None,
                    message_clsname='Pipeline',
                    message=self._log_message(step_idx),
                    logger=logger,
                    prefix=name,
                    cfg_cache=self.cfg_cache,
                    **fit_params_steps[name])

                # Extract time measurements from all sub-pipelines
                for p in cloned_transformer.pipelines.values():
                    self.fit_time += p.fit_time
                    self.config_time += p.config_time

            else:
                start = timeit.default_timer()

                Xt, fitted_transformer = _fit_transform_one(
                    cloned_transformer, Xt, y, None,
                    message_clsname='Pipeline',
                    message=self._log_message(step_idx),
                    **fit_params_steps[name])

                self.fit_time += timeit.default_timer() - start

            # Replace the transformer of the step with the fitted
            # transformer. This is necessary when loading the transformer
            # from the cache.
            self.steps[step_idx] = (name, fitted_transformer)
        if self._final_estimator == 'passthrough':
            return Xt, {}
        return Xt, fit_params_steps[self.steps[-1][0]]

    def fit(self,
            X: np.ndarray,
            y: np.ndarray = None,
            logger: ProcessLogger = None,
            prefix: str = None,
            **fit_params: dict):
        if self.configuration is None and self.cfg_cache is None:
            raise ValueError(
                'Pipeline is not configured yet. Either call set_hyperparameters or provide a ConfigGenerator')

        Xt, fit_params = self._fit(X, y, logger=logger, prefix=prefix, **fit_params)
        with _print_elapsed_time('Pipeline',
                                 self._log_message(len(self.steps) - 1)):
            if self._final_estimator != 'passthrough':

                # Configure estimator on the fly if necessary
                if self.configuration is None:
                    config = self._get_config_for_step(len(self.steps) - 1, prefix, self.steps[-1][0], logger)
                    self._final_estimator.set_hyperparameters(configuration=config.get_dictionary())

                self._final_estimator.fit(Xt, y, **fit_params)
            else:
                raise NotImplementedError('passthrough pipelines are currently not supported')
        return self

    def _get_config_for_step(self, idx: int, prefix: str, name: str,
                             logger: ProcessLogger) -> Configuration:
        start = timeit.default_timer()

        estimator = self.get_step(name)
        cfg_key = self.cfg_keys[idx]
        if isinstance(estimator, SubPipeline):
            config, cfg_key = ConfigurationSpace().get_default_configuration(), (0, 0)
        else:
            config, cfg_key = self.cfg_cache.sample_configuration(cfg_key=cfg_key)

        intermediate = PartialConfig(cfg_key, config, name, None)
        logger.new_step(prefixed_name(prefix, name), intermediate)

        self.config_time += timeit.default_timer() - start
        return config

    def set_hyperparameters(self, configuration: dict, init_params=None):
        self.configuration = configuration

        for node_idx, (node_name, node) in enumerate(self.steps):
            sub_configuration_space = node.get_hyperparameter_search_space()
            sub_config_dict = {}
            for param in configuration:
                if param.startswith('{}:'.format(node_name)):
                    value = configuration[param]
                    new_name = param.replace('{}:'.format(node_name), '', 1)
                    sub_config_dict[new_name] = value

            sub_configuration = Configuration(sub_configuration_space, values=sub_config_dict)

            if init_params is not None:
                sub_init_params_dict = {}
                for param in init_params:
                    if param.startswith('{}:'.format(node_name)):
                        value = init_params[param]
                        new_name = param.replace('{}:'.format(node_name), '', 1)
                        sub_init_params_dict[new_name] = value
            else:
                sub_init_params_dict = None

            if isinstance(node, (ComponentChoice, EstimatorComponent)):
                node.set_hyperparameters(configuration=sub_configuration.get_dictionary(),
                                         init_params=sub_init_params_dict)
            else:
                raise NotImplementedError('Not supported yet!')

        return self

    def get_hyperparameter_search_space(self, mf: MetaFeatures) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        for name, step in self.steps:
            step_configuration_space = step.get_hyperparameter_search_space(mf=mf)
            cs.add_configuration_space(name, step_configuration_space)
        return cs

    def items(self):
        return self.steps_.items()

    def as_list(self) -> List[Tuple[str, Union[str, List]]]:
        steps = []
        for name, step in self.steps:
            steps.append((name, step.serialize()))
        return steps

    @staticmethod
    def from_list(steps: List[Tuple[str, Union[str, List]]]) -> 'FlexiblePipeline':
        def __load(sub_steps: List[Tuple[str, Union[str, List]]]) -> List[Tuple[str, EstimatorComponent]]:
            d = []
            for name, value in sub_steps:
                if type(value) == str:
                    # TODO kwargs for __init__ not loaded
                    d.append((name, util.get_object(value)))
                elif type(value) == list:
                    ls = []
                    for sub_name, sub_value in value:
                        ls.append(__load(sub_value))
                    d.append((name, SubPipeline(ls)))
                else:
                    raise ValueError('Unable to handle type {}'.format(type(value)))
            return d

        ds = __load(steps)
        return FlexiblePipeline(ds)

    def __lt__(self, other: 'FlexiblePipeline'):
        s1 = tuple(e.name() for e in self.steps_.values())
        s2 = tuple(e.name() for e in other.steps_.values())
        return s1 < s2

    def __copy__(self):
        return FlexiblePipeline(clone(self.steps, safe=False), self.configuration, self.cfg_cache, self.cfg_keys)


class SubPipeline(EstimatorComponent):

    def __init__(self, sub_wfs: List[List[Tuple[str, EstimatorComponent]]]):
        self.pipelines: Dict[str, FlexiblePipeline] = {}

        # TODO cfg_keys missing
        ls = list(map(lambda wf: FlexiblePipeline(wf), sub_wfs))
        for idx, wf in enumerate(sorted(ls)):
            self.pipelines['pipeline_{}'.format(idx)] = wf

    def fit(self, X, y=None, cfg_cache: ConfigCache = None, logger: ProcessLogger = None, prefix: str = None,
            **fit_params):
        for p_name, pipeline in self.pipelines.items():
            if cfg_cache is not None:
                pipeline.cfg_cache = cfg_cache

            p_prefix = prefixed_name(prefix, p_name)
            # noinspection PyTypeChecker
            pipeline.fit(X, y, prefix=p_prefix, **fit_params)
        return self

    # noinspection PyPep8Naming
    def transform(self, X: np.ndarray):
        X_transformed = X
        for name, pipeline in self.pipelines.items():
            y_pred = pipeline.predict(X)
            X_transformed = np.hstack((X_transformed, np.reshape(y_pred, (-1, 1))))

        return X_transformed

    def set_hyperparameters(self, configuration: dict = None, init_params=None):
        if configuration is None or len(configuration.keys()) == 0:
            return

        for node_name, pipeline in self.pipelines.items():
            sub_config_dict = {}
            for param in configuration:
                if param.startswith('{}:'.format(node_name)):
                    value = configuration[param]
                    new_name = param.replace('{}:'.format(node_name), '', 1)
                    sub_config_dict[new_name] = value
            pipeline.set_hyperparameters(sub_config_dict, init_params)

    def get_hyperparameter_search_space(self, mf: MetaFeatures):
        cs = ConfigurationSpace()
        for pipeline_name, pipeline in self.pipelines.items():
            pipeline_cs = ConfigurationSpace()

            for task_name, task in pipeline.steps:
                step_configuration_space = task.get_hyperparameter_search_space(mf=mf)
                pipeline_cs.add_configuration_space(task_name, step_configuration_space)
            cs.add_configuration_space(pipeline_name, pipeline_cs)
        return cs

    def serialize(self):
        pipelines = []
        for name, p in self.pipelines.items():
            pipelines.append((name, p.as_list()))

        return pipelines

    @staticmethod
    def get_properties():
        return {}
