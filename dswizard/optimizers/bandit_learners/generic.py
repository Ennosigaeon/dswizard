import numpy as np

from dswizard.core.base_bandit_learner import BanditLearner
from dswizard.core.base_config_generator import BaseConfigGenerator
from dswizard.core.base_structure_generator import BaseStructureGenerator
from dswizard.optimizers.iterations import SuccessiveHalving


class GenericBanditLearner(BanditLearner):
    def __init__(self,
                 structure_generator: BaseStructureGenerator = None,
                 config_generator: BaseConfigGenerator = None,
                 eta: float = 3,
                 min_budget: float = 1,
                 max_budget: float = 1):
        """
        Implements a random search across the search space for comparison. Candidates are sampled at random and run on
        the maximum budget.
        :param eta: In each iteration, a complete run of sequential halving is executed. In it, after evaluating each
            configuration on the same subset size, only a fraction of 1/eta of them 'advances' to the next round. Must
            be greater or equal to 2.
        :param min_budget: budget for the evaluation
        :param max_budget: budget for the evaluation
        """
        super().__init__(structure_generator, config_generator)

        # Hyperband related stuff
        self.eta = eta
        self.min_budget = min_budget
        self.max_budget = max_budget

        # precompute some HB stuff
        self.max_iterations = -int(np.log(min_budget / max_budget) / np.log(eta)) + 1
        self.budgets = max_budget * np.power(eta, -np.linspace(self.max_iterations - 1, 0, self.max_iterations))

        self.config.update({
            'eta': eta,
            'min_budget': max_budget,
            'max_budget': max_budget,
            'budgets': self.budgets,
            'max_iterations': self.max_iterations
        })

    def _get_next_iteration(self,
                            iteration: int,
                            iteration_kwargs: dict = None) -> SuccessiveHalving:
        """
        Returns a SH iteration with only evaluations on the biggest budget
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

        return SuccessiveHalving(iteration=iteration, num_candidates=ns, budgets=self.budgets[(-s - 1):],
                                 sampler=self.structure_generator, **iteration_kwargs)
