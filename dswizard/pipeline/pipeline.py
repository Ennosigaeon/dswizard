from __future__ import annotations

import timeit
from typing import Dict, List, Tuple, TYPE_CHECKING, Optional, Any

import numpy as np
from ConfigSpace.configuration_space import Configuration, ConfigurationSpace
from sklearn.base import clone
from sklearn.pipeline import Pipeline, _fit_transform_one
from sklearn.utils import _print_elapsed_time

from dswizard.components import util
from dswizard.components.base import EstimatorComponent, HasChildComponents
from dswizard.components.util import prefixed_name, HANDLES_MULTICLASS, HANDLES_NUMERIC, HANDLES_NOMINAL, \
    HANDLES_MISSING, HANDLES_NOMINAL_CLASS
from dswizard.core.model import PartialConfig, CandidateId

if TYPE_CHECKING:
    from dswizard.core.logger import ProcessLogger
    from dswizard.core.config_cache import ConfigCache
    from dswizard.core.model import ConfigKey
    from dswizard.components.meta_features import MetaFeatures, MetaFeaturesDict


class FlexiblePipeline(Pipeline, EstimatorComponent, HasChildComponents):

    def __init__(self,
                 steps: List[Tuple[str, EstimatorComponent]],
                 configuration: Optional[dict] = None,
                 cfg_cache: Optional[ConfigCache] = None,
                 cfg_keys: Optional[List[ConfigKey]] = None):
        self.args = {'steps': [(label, util.serialize(comp)) for label, comp in steps], 'configuration': configuration}

        self.configuration = None
        self.cfg_keys = cfg_keys
        self.cfg_cache: Optional[ConfigCache] = cfg_cache
        self.cid: Optional[CandidateId] = None

        # super.__init__ has to be called after initializing all properties provided in constructor
        super().__init__(steps, verbose=False)
        self.steps: List[Tuple[str, EstimatorComponent]] = self.steps  # only for type hinting
        self.steps_ = dict(steps)
        self.configuration_space: ConfigurationSpace = self.get_hyperparameter_search_space()

        self.fit_time = 0
        self.config_time = 0

        if configuration is not None:
            self.set_hyperparameters(configuration)

    @staticmethod
    def get_properties() -> dict:
        return {'shortname': 'pipeline',
                'name': 'Flexible Pipeline',
                HANDLES_MULTICLASS: True,
                HANDLES_NUMERIC: True,
                HANDLES_NOMINAL: True,
                HANDLES_MISSING: True,
                HANDLES_NOMINAL_CLASS: True
                }

    def to_networkx(self, prefix: str = None):
        import networkx as nx
        g = nx.DiGraph()
        predecessor = None
        for name, estimator in self.steps_.items():
            name = prefixed_name(prefix, name)

            if isinstance(estimator, SubPipeline):
                split = name
                name = f'{name}_merge'

                g.add_node(split, label='split', name=name)
                g.add_node(name, label='merge')

                for p_name, p in estimator.pipelines.items():
                    prefix = prefixed_name(split, p_name)
                    h = p.to_networkx(prefix=prefix)
                    g = nx.compose(g, h)
                    g.add_edge(split, prefixed_name(prefix, p.steps[0][0]))
                    g.add_edge(prefixed_name(prefix, p.steps[-1][0]), name)
            else:
                g.add_node(name, label=estimator.name().split('.')[-1], name=name)

            if predecessor is not None:
                g.add_edge(predecessor, name)
            predecessor = name
        return g

    def get_step(self, name: str):
        step_name = name.split(':')[0]
        return self.steps_[step_name]

    def all_names(self, prefix: str = None) -> List[str]:
        res = []
        for name, component in self.steps_.items():
            res.append(prefixed_name(prefix, name))
        return res

    def _validate_steps(self):
        if len(self.steps) == 0:
            raise TypeError('Pipeline has to contain at least 1 step')
        super()._validate_steps()

    def _check_fit_params(self, **fit_params):
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
        return fit_params_steps

    def _fit(self,
             X: np.ndarray,
             y: np.ndarray = None,
             logger: ProcessLogger = None,
             prefix: str = None,
             **fit_params_steps: dict):
        # shallow copy of steps - this should really be steps_
        self.steps = list(self.steps)
        self._validate_steps()

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
        return Xt

    def fit(self,
            X: np.ndarray,
            y: np.ndarray = None,
            logger: ProcessLogger = None,
            prefix: str = None,
            **fit_params: dict):
        if self.configuration is None and self.cfg_cache is None:
            raise ValueError(
                'Pipeline is not configured yet. Either call set_hyperparameters or provide a ConfigGenerator')

        fit_params_steps = self._check_fit_params(**fit_params)
        Xt = self._fit(X, y, logger=logger, prefix=prefix, **fit_params_steps)
        with _print_elapsed_time("Pipeline", self._log_message(len(self.steps) - 1)):
            if self._final_estimator != "passthrough":
                # Configure estimator on the fly if necessary
                if self.configuration is None:
                    config = self._get_config_for_step(len(self.steps) - 1, prefix, self.steps[-1][0], logger)
                    self._final_estimator.set_hyperparameters(configuration=config.get_dictionary())

                fit_params_last_step = fit_params_steps[self.steps[-1][0]]
                self._final_estimator.fit(Xt, y, **fit_params_last_step)

        return self

    def _get_config_for_step(self, idx: int, prefix: str, name: str,
                             logger: ProcessLogger) -> Configuration:
        start = timeit.default_timer()

        cfg_key = self.cfg_keys[idx]
        config, cfg_key = self.cfg_cache.sample_configuration(cid=self.cid, name=name, cfg_key=cfg_key)

        intermediate = PartialConfig(cfg_key, config, name, None)
        logger.new_step(prefixed_name(prefix, name), intermediate)

        self.config_time += timeit.default_timer() - start
        return config

    def set_hyperparameters(self, configuration: dict = None, init_params=None):
        self.configuration = configuration
        self.set_child_hyperparameters(self.steps, configuration, init_params)
        return self

    def get_hyperparameter_search_space(self, mf: Optional[MetaFeaturesDict] = None) -> ConfigurationSpace:
        return self.get_child_hyperparameter_search_space(self.steps, mf)

    def items(self):
        return self.steps_.items()

    def get_feature_names_out(self, input_features=None):
        feature_names_out = input_features
        for _, name, transform in self._iter():
            if not hasattr(transform, "get_feature_names_out"):
                raise AttributeError(
                    "Estimator {} does not provide get_feature_names_out. "
                    "Did you mean to call pipeline[:-1].get_feature_names_out"
                    "()?".format(name)
                )
            feature_names_out = transform.get_feature_names_out(feature_names_out)
            if hasattr(transform, "predict"):
                feature_names_out = [f"{name}__{f}" for f in feature_names_out]
        return feature_names_out

    @staticmethod
    def deserialize(steps: List[str, Dict[str, Any]], **kwargs) -> 'FlexiblePipeline':
        steps_ = []
        for name, value in steps:
            steps_.append((name, util.deserialize(**value)))
        return FlexiblePipeline(steps_, **kwargs)

    def __lt__(self, other: 'FlexiblePipeline'):
        s1 = tuple(e.name() for e in self.steps_.values())
        s2 = tuple(e.name() for e in other.steps_.values())
        return s1 < s2

    def __copy__(self):
        return FlexiblePipeline(clone(self.steps, safe=False), self.configuration, self.cfg_cache, self.cfg_keys)
