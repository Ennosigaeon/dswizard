import numpy as np

from dswizard.core.base_iteration import BaseIteration


class SuccessiveHalving(BaseIteration):

    def _advance_to_next_stage(self, losses: np.ndarray) -> np.ndarray:
        """
        SuccessiveHalving simply continues the best based on the current loss.
        """
        ranks = np.argsort(np.argsort(losses))
        return ranks < self.num_candidates[self.stage]
