from typing import Optional

import numpy as np
from ConfigSpace import CategoricalHyperparameter
from ConfigSpace.configuration_space import Configuration
from ConfigSpace.hyperparameters import NumericalHyperparameter

from dswizard.core.model import CandidateId, ConfigKey, PartialConfig
from dswizard.core.base_config_generator import BaseConfigGenerator


class RandomSampling(BaseConfigGenerator):
    """
    class to implement random sampling from a ConfigSpace
    """

    def sample_config(self, cid: Optional[CandidateId] = None, cfg_key: Optional[ConfigKey] = None,
                      name: Optional[str] = None, default: bool = False) -> Configuration:
        if self.configspace is None:
            raise ValueError('No configuration space provided. Call set_config_space(ConfigurationSpace) first.')

        if default:
            config = self.configspace.get_default_configuration()
            config.origin = 'Default'
        else:
            config = self.configspace.sample_configuration()
            config.origin = 'Random Search'

        self._record_explanation(cid, cfg_key, name, config)
        return config

    def _record_explanation(self, cid: CandidateId, cfg_key: ConfigKey, name: str, config: Configuration):
        self.explanations[cid.external_name] = {
            'candidates': [PartialConfig(cfg_key, config, name, None)],
            'loss': [0.5],
            'marginalization': self._compute_marginalization()
        }

    def _compute_marginalization(self):
        res = {}
        for hp in self.configspace.get_hyperparameters():
            if isinstance(hp, CategoricalHyperparameter):
                s = np.arange(0, len(hp.choices))
            elif isinstance(hp, NumericalHyperparameter):
                if hp.log:
                    s = np.geomspace(hp.lower, hp.upper, num=10)
                else:
                    s = np.linspace(hp.lower, hp.upper, num=10)
            else:
                raise ValueError("Parameter {} of type {} not supported.".format(hp.name, type(hp)))

            res[hp.name] = {'random': np.vstack((s, np.ones(s.shape) * 0.5)).T.tolist()}

        return res
