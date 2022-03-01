from __future__ import annotations

import timeit
from typing import Dict, List, Tuple, TYPE_CHECKING, Optional, Any

import numpy as np
from ConfigSpace.configuration_space import Configuration
from sklearn.base import clone
from sklearn.pipeline import _fit_transform_one
from sklearn.utils import _print_elapsed_time

from dswizard.components import util
from dswizard.components.base import EstimatorComponent
from dswizard.components.pipeline import ConfigurablePipeline
from dswizard.components.util import prefixed_name
from dswizard.core.model import PartialConfig, CandidateId

if TYPE_CHECKING:
    from dswizard.core.logger import ProcessLogger
    from dswizard.core.config_cache import ConfigCache
    from dswizard.core.model import ConfigKey


class FlexiblePipeline(ConfigurablePipeline):

    def __init__(self,
                 steps: List[Tuple[str, EstimatorComponent]],
                 configuration: Optional[Dict] = None,
                 cfg_cache: Optional[ConfigCache] = None,
                 cfg_keys: Optional[List[ConfigKey]] = None):
        self.cfg_keys = cfg_keys
        self.cfg_cache: Optional[ConfigCache] = cfg_cache
        self.cid: Optional[CandidateId] = None

        # super.__init__ has to be called after initializing all properties provided in constructor
        super().__init__(steps, configuration)

    def get_step(self, name: str):
        step_name = name.split(':')[0]
        try:
            return self.steps_[step_name]
        except KeyError:
            # Not sure why this happens but try again with full name
            return self.steps_[name]

    def all_names(self, prefix: str = None) -> List[str]:
        res = []
        for name, component in self.steps_.items():
            res.append(prefixed_name(prefix, name))
        return res

    def _fit(self,
             X: np.ndarray,
             y: np.ndarray = None,
             logger: ProcessLogger = None,
             prefix: str = None,
             **fit_params_steps: Dict):
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
            **fit_params: Dict):
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

    @staticmethod
    def deserialize(steps: List[str, Dict[str, Any]], **kwargs) -> 'FlexiblePipeline':
        steps_ = []
        for name, value in steps:
            steps_.append((name, util.deserialize(**value)))
        return FlexiblePipeline(steps_, **kwargs)

    def __copy__(self):
        return FlexiblePipeline(clone(self.steps, safe=False), self.configuration, self.cfg_cache, self.cfg_keys)
