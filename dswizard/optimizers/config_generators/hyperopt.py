from __future__ import annotations

import warnings
from typing import Dict, List, Optional

import ConfigSpace
import ConfigSpace.hyperparameters
import ConfigSpace.util
import numpy as np
import scipy.stats as sps
import statsmodels.api as sm
from ConfigSpace.configuration_space import ConfigurationSpace, Configuration

from dswizard.core.base_config_generator import BaseConfigGenerator
from dswizard.core.model import StatusType


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
                 random_fraction: float = 1 / 3,
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

    def sample_config(self, default: bool = False) -> Configuration:
        try:
            sample = None
            if len(self.kde.losses) == 0 or default:
                sample = self.configspace.get_default_configuration()
            elif self.kde.is_trained():
                sample = self._draw_sample()

            if sample is None:
                sample = self.configspace.sample_configuration()
        except:
            sample = self.configspace.sample_configuration()

        return sample

    def _draw_sample(self) -> Optional[Configuration]:
        best_ei = np.inf
        best_vector = None

        l = self.kde.good_kde().pdf
        g = self.kde.bad_kde().pdf

        kde_good = self.kde.good_kde()
        kde_bad = self.kde.bad_kde()

        for i in range(self.num_samples):
            idx = np.random.randint(0, len(kde_good.data))
            datum = kde_good.data[idx]
            vector = []

            for m, bw, t in zip(datum, kde_good.bw, self.kde.vartypes):
                bw = max(bw, self.min_bandwidth)
                if t == 0:
                    bw = self.bw_factor * bw
                    try:
                        vector.append(sps.truncnorm.rvs(-m / bw, (1 - m) / bw, loc=m, scale=bw))
                    except:
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
                ei = max(1e-32, g(vector)) / max(l(vector), 1e-32)

                if not np.isfinite(ei):
                    # right now, this happens because a KDE does not contain all values for a categorical
                    # parameter this cannot be fixed with the statsmodels KDE, so for now, we are just going to
                    # evaluate this one if the good_kde has a finite value, i.e. there is no config with that
                    # value in the bad kde, so it shouldn't be terrible.
                    if np.isfinite(l(vector)):
                        best_vector = vector
                        break

            if ei < best_ei:
                best_ei = ei
                best_vector = vector

        if best_vector is None:
            return None

        for i, hp_value in enumerate(best_vector):
            if isinstance(self.configspace.get_hyperparameter(self.configspace.get_hyperparameter_by_idx(i)),
                          ConfigSpace.hyperparameters.CategoricalHyperparameter):
                best_vector[i] = int(np.rint(best_vector[i]))
        # noinspection PyTypeChecker
        config = ConfigSpace.Configuration(self.configspace, vector=best_vector)
        try:
            config.is_valid_configuration()
            return config
        except ValueError:
            self.register_result(config, self.worst_score, StatusType.CRASHED)
            return self.configspace.sample_configuration()

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
        self.kde.configs.append(config.get_array())

        min_points_in_model = max(len(self.configspace.get_hyperparameters()) + 1, self.min_points_in_model)
        # skip model building if not enough points are available
        if len(self.kde.losses) < min_points_in_model:
            return

        train_losses = np.array(self.kde.losses)

        n_good = max(min_points_in_model, (self.top_n_percent * train_losses.shape[0]) // 100)
        # TODO use this??? Is bad trainings data complete set without n_good?
        # n_bad = max(self.min_points_in_model, train_configs.shape[0] - n_good)
        n_bad = max(min_points_in_model, ((100 - self.top_n_percent) * train_losses.shape[0]) // 100)

        idx = np.argsort(train_losses)
        train_configs = np.array(self.kde.configs)
        train_data_good = np.array(train_configs[idx[:n_good]])
        train_data_bad = np.array(train_configs[idx[-n_bad:]])

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
            'bad': bad_kde
        }

    # noinspection PyMethodMayBeStatic
    def _build_kde_wrapper(self, configspace: ConfigurationSpace) -> KdeWrapper:
        kde_vartypes = ''
        vartypes = []

        for h in configspace.get_hyperparameters():
            if hasattr(h, 'sequence'):
                raise RuntimeError('This version on dswizard does not support ordinal hyperparameters. '
                                   'Please encode {} as an integer parameter!'.format(h.name))
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
