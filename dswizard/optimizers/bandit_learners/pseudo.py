import math
from typing import Dict

from dswizard.core.base_bandit_learner import BanditLearner
from dswizard.core.base_iteration import BaseIteration
from dswizard.optimizers.iterations.pseudo import PseudoIteration


class PseudoBandit(BanditLearner):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.max_iterations = math.inf

    def _get_next_iteration(self, iteration: int, iteration_kwargs: Dict) -> BaseIteration:
        if iteration_kwargs is None:
            iteration_kwargs = {}
        return PseudoIteration(iteration, **iteration_kwargs)
