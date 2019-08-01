import numpy as np

from hpbandster.core.master import Master
from hpbandster.core.base_structure_generator import BaseStructureGenerator
from hpbandster.optimizers.config_generators import RandomSampling
from hpbandster.optimizers.iterations import SuccessiveHalving


class HyperBand(Master):
    def __init__(self,
                 structure: BaseStructureGenerator = None,
                 eta: float = 3,
                 min_budget: float = 0.01,
                 max_budget: float = 1,
                 timeout: float = None,
                 **kwargs):
        """
        Hyperband implements hyperparameter optimization by sampling candidates at random and "trying" them first,
        running them for a specific budget. The approach is iterative, promising candidates are run for a longer time,
        increasing the fidelity for their performance. While this is a very efficient racing approach, random sampling
        makes no use of the knowledge gained about the candidates during optimization.
        :param structure: a method for generating pipeline structures
        :param eta: In each iteration, a complete run of sequential halving is executed. In it, after evaluating each
            configuration on the same subset size, only a fraction of 1/eta of them 'advances' to the next round.
            Must be greater or equal to 2.
        :param min_budget: The smallest budget to consider. Needs to be positive!
        :param max_budget: the largest budget to consider. Needs to be larger than min_budget! The budgets will be
            geometrically distributed $\sim \eta^k$ for $k\in [0, 1, ... , num_subsets - 1]$.
        :param timeout: Maximum time in seconds available to evaluate a single configuration. The timout will be
            automatically adjusted to the current budget.
        :param kwargs:
        """
        structure.set_config_generator(RandomSampling())
        super().__init__(config_generator=structure, **kwargs)

        # Hyperband related stuff
        self.eta = eta
        self.min_budget = min_budget
        self.max_budget = max_budget
        self.timeout = timeout

        # precompute some HB stuff
        self.max_iterations = -int(np.log(min_budget / max_budget) / np.log(eta)) + 1
        self.budgets = max_budget * np.power(eta, -np.linspace(self.max_iterations - 1, 0, self.max_iterations))

        self.config.update({
            'eta': eta,
            'min_budget': min_budget,
            'max_budget': max_budget,
            'budgets': self.budgets,
            'timeout': self.timeout,
            'max_iterations': self.max_iterations,
        })

    def get_next_iteration(self,
                           iteration: int,
                           iteration_kwargs: dict = None) -> SuccessiveHalving:
        """
        Hyperband uses SuccessiveHalving for each iteration. See Li et al. (2016) for reference.
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
                                 config_sampler=self.config_generator, **iteration_kwargs)
