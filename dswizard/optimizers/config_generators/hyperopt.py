from __future__ import annotations

import warnings
from typing import Dict, List, Optional, Tuple

import ConfigSpace
import ConfigSpace.hyperparameters
import ConfigSpace.util
import numpy as np
import scipy.stats as sps
import statsmodels.api as sm
from ConfigSpace import CategoricalHyperparameter
from ConfigSpace.configuration_space import ConfigurationSpace, Configuration
from ConfigSpace.hyperparameters import NumericalHyperparameter

from dswizard.core.base_config_generator import BaseConfigGenerator
from dswizard.core.model import PartialConfig, ConfigKey, CandidateId, StatusType


class KdeWrapper:

    def __init__(self, kde_vartypes: str, vartypes: np.ndarray):
        self.kde_vartypes: str = kde_vartypes
        self.vartypes: np.ndarray = vartypes
        self.kde_models: Dict[str, sm.nonparametric.KDEMultivariate] = {}
        self.configs: List[np.ndarray] = []
        self.losses: List[float] = []

    def is_trained(self) -> bool:
        return 'good' in self.kde_models and 'bad' in self.kde_models

    def good_kde(self):
        return self.kde_models['good']

    def bad_kde(self):
        return self.kde_models['bad']


class Hyperopt(BaseConfigGenerator):

    def __init__(self,
                 configspace: ConfigurationSpace,
                 min_points_in_model: int = 0,
                 top_n_percent: int = 15,
                 num_samples: int = 64,
                 random_fraction: float = 0.2,
                 bandwidth_factor: float = 3,
                 min_bandwidth: float = 1e-3,
                 worst_score: float = np.inf,
                 **kwargs):
        """
        Fits a kernel density estimator on the best N percent of the evaluated configurations.

        :param configspace: Configuration space object
        :param pipeline: optional pipeline associated with the ConfigurationSpace
        :param min_points_in_model: Determines the percentile of configurations that will be used as training data for
            the kernel density estimator, e.g if set to 10 the 10% best configurations will be considered for training.
        :param top_n_percent: minimum number of datapoints needed to fit a model
        :param num_samples: number of samples drawn to optimize EI via sampling
        :param random_fraction: fraction of random configurations returned
        :param bandwidth_factor: widens the bandwidth for continuous parameters for proposed points to optimize EI
        :param min_bandwidth: to keep diversity, even when all (good) samples have the same value for one of the
            parameters, a minimum bandwidth (Default: 1e-3) is used instead of zero.
        :param kwargs:
        """

        super().__init__(configspace, **kwargs)

        self.top_n_percent = top_n_percent
        self.bw_factor = bandwidth_factor
        self.min_bandwidth = min_bandwidth
        self.worst_score = worst_score

        self.min_points_in_model = min_points_in_model
        self.num_samples = num_samples
        self.random_fraction = random_fraction

        self.kde: KdeWrapper = self._build_kde_wrapper(self.configspace)

    def sample_config(self, cid: Optional[CandidateId] = None, cfg_key: Optional[ConfigKey] = None,
                      name: Optional[str] = None, default: bool = False) -> Configuration:
        try:
            if len(self.kde.losses) == 0 or default:
                config = self.configspace.get_default_configuration()
                config.origin = 'Default'
                candidates = [config] * self.num_samples
                candidates_ei = [1] * self.num_samples
            elif self.kde.is_trained() and np.random.random() > self.random_fraction:
                candidates, candidates_ei = self._sample_candidates()
                config = candidates[np.argmax(candidates_ei)]
            else:
                raise ValueError('Not fitted yet')
        except Exception:
            config = self.configspace.sample_configuration()
            config.origin = 'Random Search'
            candidates = [config] * self.num_samples
            candidates_ei = [1] * self.num_samples

        self._record_explanation(cid, cfg_key, name, candidates, candidates_ei)
        return config

    def _sample_candidates(self) -> Tuple[List[Configuration], List[float]]:
        candidates_ei = []
        candidates = []

        good = self.kde.good_kde().pdf
        bad = self.kde.bad_kde().pdf

        kde_good = self.kde.good_kde()

        while len(candidates) < self.num_samples:
            idx = np.random.randint(0, len(kde_good.data))
            datum = kde_good.data[idx]
            vector = []

            for m, bw, t, hp in zip(datum, kde_good.bw, self.kde.vartypes, self.configspace.get_hyperparameters()):
                bw = max(bw, self.min_bandwidth)
                if t == 0:
                    bw = self.bw_factor * bw
                    try:
                        lower, upper = hp._inverse_transform(hp.lower), hp._inverse_transform(hp.upper)
                        vector.append(sps.truncnorm.rvs((lower - m) / bw, (upper - m) / bw, loc=m, scale=bw))
                    except Exception:
                        pass
                else:
                    if np.random.rand() < (1 - bw):
                        vector.append(int(m))
                    else:
                        vector.append(np.random.randint(t))

            # DivideByZero in pdf evaluation if all values of categorical variable are identical
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # Calculate expected improvement
                ei = max(1e-32, good(vector)) / max(bad(vector), 1e-32)

            config = ConfigSpace.Configuration(self.configspace, vector=vector, allow_inactive_with_values=True)
            try:
                config.is_valid_configuration()

                # Remove inactive hyperparameters
                active_hp = self.configspace.get_active_hyperparameters(config)
                d = {key: value for key, value in config.get_dictionary().items() if key in active_hp}
                config = ConfigSpace.Configuration(self.configspace, d)

                config.origin = 'Hyperopt'
                candidates.append(config)
                candidates_ei.append(float(ei))
            except ValueError:
                self.register_result(config, self.worst_score, StatusType.CRASHED)

        return candidates, candidates_ei

    def _record_explanation(self, cid: CandidateId, cfg_key: ConfigKey, name: str,
                            candidates: List[Configuration], loss: List[float]):
        self.explanations[cid.external_name] = {
            'candidates': [PartialConfig(cfg_key, c, name, None) for c in candidates],
            'loss': loss,
            'marginalization': self._compute_marginalization()
        }

    def _compute_marginalization(self):
        if len(self.configspace.get_hyperparameters()) == 0 or not self.kde.is_trained():
            return {}

        support = []

        for hp in self.configspace.get_hyperparameters():
            if isinstance(hp, CategoricalHyperparameter):
                support.append(np.arange(0, len(hp.choices)))
            elif isinstance(hp, NumericalHyperparameter):
                if hp.log:
                    s = np.geomspace(hp.lower, hp.upper, num=8)
                else:
                    s = np.linspace(hp.lower, hp.upper, num=8)
                support.append(hp._inverse_transform(s))

        grid = np.meshgrid(*support)
        dimensions = [np.ravel(a) for a in grid]
        points = np.vstack(dimensions).T

        with warnings.catch_warnings():
            good = self.kde.good_kde().pdf(points)
            bad = self.kde.bad_kde().pdf(points)

        res = {}
        for hp, d in zip(self.configspace.get_hyperparameters(), dimensions):
            mar_good = []
            mar_bad = []
            mar_ei = []
            unique = np.unique(d)
            for v in np.sort(unique):
                avg_good = np.nanmean(good[d == v])
                avg_bad = np.nanmean(bad[d == v])

                v_inv = hp._transform(v)
                mar_good.append((v_inv, float(avg_good)))
                mar_bad.append((v_inv, float(avg_bad)))
                mar_ei.append((v_inv, float(avg_good / avg_bad)))
            res[hp.name] = {'good': mar_good, 'bad': mar_bad, 'ei': mar_ei}
            # If necessary, 'bad' can be recorded accordingly

        return res

    def register_result(self, config: Configuration, loss: float, status: StatusType,
                        update_model: bool = True, **kwargs) -> None:
        super().register_result(config, loss, status)
        # noinspection PyUnresolvedReferences
        actual_size = config.get_array().size
        if actual_size != self.expected_size:
            return

        if loss is None or not np.isfinite(loss):
            loss = self.worst_score

        self.kde.losses.append(loss)
        # noinspection PyTypeChecker
        self.kde.configs.append(config.get_array())

        min_points_in_model = max(int(1.5 * len(self.configspace.get_hyperparameters())) + 1, self.min_points_in_model)
        # skip model building if not enough points are available
        if len(self.kde.losses) < min_points_in_model:
            return

        train_losses = np.array(self.kde.losses)

        n_good = max(min_points_in_model, (self.top_n_percent * train_losses.shape[0]) // 100)
        n_bad = max(min_points_in_model, ((100 - self.top_n_percent) * train_losses.shape[0]) // 100)

        idx = np.argsort(train_losses)
        train_configs = np.array(self.kde.configs)
        train_data_good = self._fix_identical_cat_input(train_configs[idx[:n_good]])
        train_data_bad = self._fix_identical_cat_input(train_configs[idx[-n_bad:]])

        train_data_good = self._impute_conditional_data(train_data_good, self.kde.vartypes)
        train_data_bad = self._impute_conditional_data(train_data_bad, self.kde.vartypes)

        if train_data_good.shape[0] <= train_data_good.shape[1]:
            return
        if train_data_bad.shape[0] <= train_data_bad.shape[1]:
            return

        # more expensive cross-validation method
        # bw_estimation = 'cv_ls'

        # quick rule of thumb
        bw_estimation = 'normal_reference'

        bad_kde = sm.nonparametric.KDEMultivariate(data=train_data_bad, var_type=self.kde.kde_vartypes,
                                                   bw=bw_estimation)
        good_kde = sm.nonparametric.KDEMultivariate(data=train_data_good, var_type=self.kde.kde_vartypes,
                                                    bw=bw_estimation)

        bad_kde.bw = np.clip(bad_kde.bw, self.min_bandwidth, None)
        good_kde.bw = np.clip(good_kde.bw, self.min_bandwidth, None)

        self.kde.kde_models = {
            'good': good_kde,
            'bad': bad_kde,
        }

    def _fix_identical_cat_input(self, train_data: np.ndarray):
        # KDE can not handle if all categorical values are identical. Add default configuration with adapted categorical
        # values to prevent division by 0

        cat_idx = self.kde.vartypes > 0
        identical = np.min(train_data[:, cat_idx], axis=0) == np.max(train_data[:, cat_idx], axis=0)

        if np.any(identical):
            additional = self.configspace.get_default_configuration().get_array()
            for idx in np.argwhere(cat_idx)[identical]:
                idx = idx[0]
                value = train_data[0, idx]
                if value == additional[idx]:
                    additional[idx] = 1 if value == 0 else 0
            train_data = np.vstack((train_data, additional))

        return train_data

    # noinspection PyMethodMayBeStatic
    def _build_kde_wrapper(self, configspace: ConfigurationSpace) -> KdeWrapper:
        kde_vartypes = ''
        vartypes = []

        for h in configspace.get_hyperparameters():
            if hasattr(h, 'sequence'):
                raise RuntimeError('This version on dswizard does not support ordinal hyperparameters. '
                                   f'Please encode {h.name} as an integer parameter!')
            if hasattr(h, 'choices'):
                kde_vartypes += 'u'
                vartypes += [len(h.choices)]
            else:
                kde_vartypes += 'c'
                vartypes += [0]

        vartypes = np.array(vartypes, dtype=int)
        return KdeWrapper(kde_vartypes, vartypes)

    # noinspection PyMethodMayBeStatic
    def _impute_conditional_data(self, array: np.ndarray, vartypes: np.ndarray) -> np.ndarray:
        """
        Impute all conditional (nan) values. The following steps are executed for each sample:
        1. Find all missing values
        2. For each missing value check if other samples have a value assigned
            - If yes, select one of the instances at random
            - If no, select random value from valid space, i.e. [0, 1]

        :param array:
        :return:
        """

        return_array = np.zeros_like(array)

        for i in range(array.shape[0]):
            datum = np.copy(array[i])
            nan_indices = np.argwhere(np.isnan(datum)).flatten()

            while np.any(nan_indices):
                nan_idx = nan_indices[0]
                valid_indices = np.argwhere(np.isfinite(array[:, nan_idx])).flatten()

                if len(valid_indices) > 0:
                    # pick one of them at random and overwrite all NaN values
                    row_idx = np.random.choice(valid_indices)
                    datum[nan_indices] = array[row_idx, nan_indices]

                else:
                    # no good point in the data has this value activated, so fill it with a valid but random value
                    t = vartypes[nan_idx]
                    if t == 0:
                        datum[nan_idx] = np.random.rand()
                    else:
                        datum[nan_idx] = np.random.randint(t)

                nan_indices = np.argwhere(np.isnan(datum)).flatten()
            return_array[i, :] = datum
        return return_array
