import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace

from dswizard.core.base_hpo import HPO
from dswizard.core.model import Structure
from dswizard.optimizers.config_generators.smac import SMAC as cg_smac
from dswizard.optimizers.iterations import SuccessiveHalving


class SMAC(HPO):
    def __init__(self,
                 configspace: ConfigurationSpace = None,
                 structure: Structure = None,
                 eta: float = 3,
                 min_budget: float = 0.01,
                 max_budget: float = 1,
                 timeout: float = None,
                 num_samples: int = 64):
        """
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
        :param num_samples: number of samples to optimize EI (default 64)
        """

        # TODO: Proper check for ConfigSpace object!
        if configspace is None:
            raise ValueError('You have to provide a valid ConfigSpace object')

        cg = cg_smac(
            configspace=configspace,
            structure=structure,
            num_samples=num_samples
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
            'num_samples': num_samples,
        })

    def get_next_iteration(self,
                           iteration: int,
                           iteration_kwargs: dict = None) -> SuccessiveHalving:

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
