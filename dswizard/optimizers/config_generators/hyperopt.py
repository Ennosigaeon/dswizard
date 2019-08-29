import traceback
from typing import Optional

import ConfigSpace
import ConfigSpace.hyperparameters
import ConfigSpace.util
import numpy as np
import scipy.stats as sps
import statsmodels.api as sm
from ConfigSpace.configuration_space import ConfigurationSpace, Configuration

from dswizard.components.pipeline import FlexiblePipeline
from dswizard.core.base_config_generator import BaseConfigGenerator
from dswizard.core.model import Job


class Hyperopt(BaseConfigGenerator):
    def __init__(self,
                 configspace: ConfigurationSpace,
                 pipeline: FlexiblePipeline = None,
                 min_points_in_model: int = None,
                 top_n_percent: int = 15,
                 num_samples: int = 64,
                 random_fraction: float = 1 / 3,
                 bandwidth_factor: float = 3,
                 min_bandwidth: float = 1e-3,
                 **kwargs):
        """
        Fits for each given budget a kernel density estimator on the best N percent of the evaluated configurations on
        this budget.
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

        super().__init__(configspace, pipeline, **kwargs)
        self.top_n_percent = top_n_percent
        self.bw_factor = bandwidth_factor
        self.min_bandwidth = min_bandwidth

        self.min_points_in_model = min_points_in_model
        if min_points_in_model is None:
            self.min_points_in_model = len(self.configspace.get_hyperparameters()) + 1

        if self.min_points_in_model < len(self.configspace.get_hyperparameters()) + 1:
            self.logger.warning('Invalid min_points_in_model value. Setting it to {}'.format(
                len(self.configspace.get_hyperparameters()) + 1))
            self.min_points_in_model = len(self.configspace.get_hyperparameters()) + 1

        self.num_samples = num_samples
        self.random_fraction = random_fraction

        self.kde_vartypes = ""
        self.vartypes = []

        for h in self.configspace.get_hyperparameters():
            if hasattr(h, 'sequence'):
                raise RuntimeError('This version on BOHB does not support ordinal hyperparameters. '
                                   'Please encode {} as an integer parameter!'.format(h.name))
            if hasattr(h, 'choices'):
                self.kde_vartypes += 'u'
                self.vartypes += [len(h.choices)]
            else:
                self.kde_vartypes += 'c'
                self.vartypes += [0]

        self.vartypes = np.array(self.vartypes, dtype=int)

        # store precomputed probabilities for the categorical parameters
        self.cat_probs = []

        self.configs = dict()
        self.losses = dict()
        self.good_config_rankings = dict()
        self.kde_models = dict()

    def largest_budget_with_model(self) -> float:
        if len(self.kde_models) == 0:
            return -float('inf')
        return max(self.kde_models.keys())

    def get_config(self, budget: float = None) -> Configuration:
        if self.configspace is None:
            raise ValueError('No configuration space provided. Call set_config_space(ConfigurationSpace) first.')

        self.logger.debug('start sampling a new configuration.')

        sample: Optional[Configuration] = None

        # If no model is available, sample from prior
        # also mix in a fraction of random configs
        if len(self.kde_models.keys()) == 0 or np.random.rand() < self.random_fraction:
            sample = self.configspace.sample_configuration()

        best = np.inf
        best_vector = None

        if sample is None:
            try:

                # sample from largest budget
                budget = max(self.kde_models.keys())

                l = self.kde_models[budget]['good'].pdf
                g = self.kde_models[budget]['bad'].pdf

                minimize_me = lambda x: max(1e-32, g(x)) / max(l(x), 1e-32)

                kde_good = self.kde_models[budget]['good']
                kde_bad = self.kde_models[budget]['bad']

                for i in range(self.num_samples):
                    idx = np.random.randint(0, len(kde_good.data))
                    datum = kde_good.data[idx]
                    vector = []

                    for m, bw, t in zip(datum, kde_good.bw, self.vartypes):
                        bw = max(bw, self.min_bandwidth)
                        if t == 0:
                            bw = self.bw_factor * bw
                            try:
                                vector.append(sps.truncnorm.rvs(-m / bw, (1 - m) / bw, loc=m, scale=bw))
                            except:
                                self.logger.warning(
                                    'Truncated Normal failed for:\ndatum={}\nbandwidth={}\nfor entry with value {}'.format(
                                        datum, kde_good.bw, m))
                                self.logger.warning('data in the KDE:\n{}'.format(kde_good.data))
                        else:
                            if np.random.rand() < (1 - bw):
                                vector.append(int(m))
                            else:
                                vector.append(np.random.randint(t))
                    val = minimize_me(vector)

                    if not np.isfinite(val):
                        self.logger.warning('sampled vector: {} has EI value {}'.format(vector, val))
                        self.logger.warning('data in the KDEs:\n{}\n{}'.format(kde_good.data, kde_bad.data))
                        self.logger.warning('bandwidth of the KDEs:\n{}\n{}'.format(kde_good.bw, kde_bad.bw))
                        self.logger.warning('l(x) = {}'.format(l(vector)))
                        self.logger.warning('g(x) = {}'.format(g(vector)))

                        # right now, this happens because a KDE does not contain all values for a categorical parameter
                        # this cannot be fixed with the statsmodels KDE, so for now, we are just going to evaluate this
                        # one if the good_kde has a finite value, i.e. there is no config with that value in the bad
                        # kde, so it shouldn't be terrible.
                        if np.isfinite(l(vector)):
                            best_vector = vector
                            break

                    if val < best:
                        best = val
                        best_vector = vector

                if best_vector is None:
                    self.logger.debug(
                        'Sampling based optimization with {} samples failed -> using random configuration'.format(
                            self.num_samples))
                    sample = self.configspace.sample_configuration()
                else:
                    self.logger.debug(
                        'best_vector: {}, {}, {}, {}'.format(best_vector, best, l(best_vector), g(best_vector)))
                    for i, hp_value in enumerate(best_vector):
                        if isinstance(
                                self.configspace.get_hyperparameter(
                                    self.configspace.get_hyperparameter_by_idx(i)
                                ),
                                ConfigSpace.hyperparameters.CategoricalHyperparameter
                        ):
                            best_vector[i] = int(np.rint(best_vector[i]))
                    # noinspection PyTypeChecker
                    sample = ConfigSpace.Configuration(self.configspace, vector=best_vector)

                    try:
                        sample = ConfigSpace.util.deactivate_inactive_hyperparameters(
                            configuration_space=self.configspace,
                            configuration=sample.get_dictionary()
                        )
                    except Exception as e:
                        self.logger.warning(("=" * 50 + "\n") * 3 +
                                            'Error converting configuration:\n{}'.format(sample.get_dictionary()) +
                                            '\n here is a traceback:' +
                                            traceback.format_exc())
                        raise e
            except:
                self.logger.warning(
                    'Sampling based optimization with {} samples failed\n {} \nUsing random configuration'.format(
                        self.num_samples, traceback.format_exc()))
                sample = self.configspace.sample_configuration()

        try:
            sample = ConfigSpace.util.deactivate_inactive_hyperparameters(
                configuration_space=self.configspace,
                configuration=sample.get_dictionary()
            )
        except Exception as e:
            self.logger.warning('Error ({}) converting configuration: {} -> '
                                'using random configuration!'.format(e, sample))
            sample = self.configspace.sample_configuration()
        self.logger.debug('done sampling a new configuration.')
        return sample

    def get_config_for_step(self, step: str, budget: float = None) -> Configuration:
        raise NotImplementedError('Hyperopt does not support JIT configuration')

    def register_result(self,
                        job: Job,
                        update_model: bool = True) -> None:
        """
        function to register finished runs

        Every time a run has finished, this function should be called to register it with the result logger. If
        overwritten, make sure to call this method from the base class to ensure proper logging.
        :param job: contains all the info about the run
        :param update_model:
        :return:
        """

        super().register_result(job)

        if job.result is None:
            # One could skip crashed results, but we decided to
            # assign a +inf loss and count them as bad configurations
            loss = np.inf
        else:
            # same for non numeric losses.
            # Note that this means losses of minus infinity will count as bad!
            loss = job.result.loss if job.result.loss is not None and np.isfinite(job.result.loss) else np.inf

        budget = job.budget

        if budget not in self.configs.keys():
            self.configs[budget] = []
            self.losses[budget] = []

        # skip model building if we already have a bigger model
        if max(list(self.kde_models.keys()) + [-np.inf]) > budget:
            return

        # We want to get a numerical representation of the configuration in the original space

        conf = ConfigSpace.Configuration(self.configspace, job.config)
        self.configs[budget].append(conf.get_array())
        self.losses[budget].append(loss)

        # skip model building:
        # a) if not enough points are available
        if len(self.configs[budget]) <= self.min_points_in_model - 1:
            self.logger.debug("Only {} run(s) for budget {} available, need more than {} -> can't build model!".format(
                len(self.configs[budget]), budget, self.min_points_in_model + 1))
            return

        # b) during warm starting when we feed previous results in and only update once
        if not update_model:
            return

        train_configs = np.array(self.configs[budget])
        train_losses = np.array(self.losses[budget])

        n_good = max(self.min_points_in_model, (self.top_n_percent * train_configs.shape[0]) // 100)
        # TODO use this??? Is bad trainings data complete set without n_good?
        # n_bad = max(self.min_points_in_model, train_configs.shape[0] - n_good)
        n_bad = max(self.min_points_in_model, ((100 - self.top_n_percent) * train_configs.shape[0]) // 100)

        # Refit KDE for the current budget
        idx = np.argsort(train_losses)

        train_data_good = self._impute_conditional_data(train_configs[idx[:n_good]])
        train_data_bad = self._impute_conditional_data(train_configs[idx[-n_bad:]])

        if train_data_good.shape[0] <= train_data_good.shape[1]:
            return
        if train_data_bad.shape[0] <= train_data_bad.shape[1]:
            return

        # more expensive cross-validation method
        # bw_estimation = 'cv_ls'

        # quick rule of thumb
        bw_estimation = 'normal_reference'

        bad_kde = sm.nonparametric.KDEMultivariate(data=train_data_bad, var_type=self.kde_vartypes, bw=bw_estimation)
        good_kde = sm.nonparametric.KDEMultivariate(data=train_data_good, var_type=self.kde_vartypes, bw=bw_estimation)

        bad_kde.bw = np.clip(bad_kde.bw, self.min_bandwidth, None)
        good_kde.bw = np.clip(good_kde.bw, self.min_bandwidth, None)

        self.kde_models[budget] = {
            'good': good_kde,
            'bad': bad_kde
        }

        # update probabilities for the categorical parameters for later sampling
        self.logger.debug(
            'done building a new model for budget {} based on {}/{} split. Best loss for this budget:{}'.format(
                budget, n_good, n_bad, np.min(train_losses)))

    def _impute_conditional_data(self, array):
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
                    t = self.vartypes[nan_idx]
                    if t == 0:
                        datum[nan_idx] = np.random.rand()
                    else:
                        datum[nan_idx] = np.random.randint(t)

                nan_indices = np.argwhere(np.isnan(datum)).flatten()
            return_array[i, :] = datum
        return return_array
