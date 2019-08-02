import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace

from core.base_hpo import HPO
from hpbandster.core.model import Structure
from hpbandster.optimizers.config_generators import Hyperopt
from hpbandster.optimizers.iterations import SuccessiveHalving


class BOHB(HPO):
    def __init__(self,
                 configspace: ConfigurationSpace = None,
                 structure: Structure = None,
                 eta: float = 3,
                 min_budget: float = 0.01,
                 max_budget: float = 1,
                 timeout: float = None,
                 min_points_in_model: int = None,
                 top_n_percent: int = 15,
                 num_samples: int = 64,
                 random_fraction: float = 1 / 3,
                 bandwidth_factor: float = 3,
                 min_bandwidth: float = 1e-3):
        """
        BOHB performs robust and efficient hyperparameter optimization at scale by combining the speed of Hyperband
        searches with the guidance and guarantees of convergence of Bayesian Optimization. Instead of sampling new
        configurations at random, BOHB uses kernel density estimators to select promising candidates.

        For reference: ::

            @InProceedings{falkner-icml-18,
              title =        {{BOHB}: Robust and Efficient Hyperparameter Optimization at Scale},
              author =       {Falkner, Stefan and Klein, Aaron and Hutter, Frank},
              booktitle =    {Proceedings of the 35th International Conference on Machine Learning},
              pages =        {1436--1445},
              year =         {2018},
            }

        :param configspace: valid representation of the search space
        :param structure: optional structure associated with the ConfigurationSpace
        :param eta: In each iteration, a complete run of sequential halving is executed. In it, after evaluating each
            configuration on the same subset size, only a fraction of 1/eta of them 'advances' to the next round. Must
            be greater or equal to 2.
        :param min_budget: The smallest budget to consider. Needs to be positive!
        :param max_budget: The largest budget to consider. Needs to be larger than min_budget! The budgets will be
         geometrically distributed :math:`a^2 + b^2 = c^2 \sim \eta^k` for :math:`k\in [0, 1, ... , num\_subsets - 1]`.
        :param timeout: Maximum time in seconds available to evaluate a single configuration. The timout will be
            automatically adjusted to the current budget.
        :param min_points_in_model: number of observations to start building a KDE. Default 'None' means dim+1, the
            bare minimum.
        :param top_n_percent: percentage ( between 1 and 99, default 15) of the observations that are considered good.
        :param num_samples: number of samples to optimize EI (default 64)
        :param random_fraction: fraction of purely random configurations that are sampled from the prior without the
            model.
        :param bandwidth_factor: to encourage diversity, the points proposed to optimize EI, are sampled from a
            'widened' KDE where the bandwidth is multiplied by this factor (default: 3)
        :param min_bandwidth: to keep diversity, even when all (good) samples have the same value for one of the
            parameters, a minimum bandwidth (Default: 1e-3) is used instead of zero.
        """

        # TODO: Proper check for ConfigSpace object!
        if configspace is None:
            raise ValueError('You have to provide a valid ConfigSpace object')

        cg = Hyperopt(
            configspace=configspace,
            structure=structure,
            min_points_in_model=min_points_in_model,
            top_n_percent=top_n_percent,
            num_samples=num_samples,
            random_fraction=random_fraction,
            bandwidth_factor=bandwidth_factor,
            min_bandwidth=min_bandwidth
        )

        super().__init__(cg)

        # Hyperband related stuff
        self.eta = eta
        self.min_budget = min_budget
        self.max_budget = max_budget

        # precompute some HB stuff
        self.max_iterations = -int(np.log(min_budget / max_budget) / np.log(eta)) + 1
        self.budgets = max_budget * np.power(eta, -np.linspace(self.max_iterations - 1, 0, self.max_iterations))
        self.timeout = timeout

        self.config.update({
            'eta': eta,
            'min_budget': min_budget,
            'max_budget': max_budget,
            'budgets': self.budgets,
            'timeout': self.timeout,
            'max_iterations': self.max_iterations,
            'min_points_in_model': min_points_in_model,
            'top_n_percent': top_n_percent,
            'num_samples': num_samples,
            'random_fraction': random_fraction,
            'bandwidth_factor': bandwidth_factor,
            'min_bandwidth': min_bandwidth
        })

    def get_next_iteration(self,
                           iteration: int,
                           iteration_kwargs: dict = None) -> SuccessiveHalving:
        """
        BOHB uses (just like Hyperband) SuccessiveHalving for each iteration. See Li et al. (2016) for reference.
        :param iteration: the index of the iteration to be instantiated
        :param iteration_kwargs: default
        :return: the SuccessiveHalving iteration with the corresponding number of configurations
        """

        if iteration_kwargs is None:
            iteration_kwargs = {}
        # number of 'SH rungs'
        s = self.max_iterations - 1 - (iteration % self.max_iterations)
        # number of configurations in that bracket
        n0 = int(np.floor(self.max_iterations / (s + 1)) * self.eta ** s)
        ns = [max(int(n0 * (self.eta ** (-i))), 1) for i in range(s + 1)]

        return SuccessiveHalving(HPB_iter=iteration, num_configs=ns, budgets=self.budgets[(-s - 1):],
                                 timeout=self.timeout, config_sampler=self.config_generator,
                                 **iteration_kwargs)
