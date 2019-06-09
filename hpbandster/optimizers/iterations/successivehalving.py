import numpy as np
from typing import List, Tuple

from hpbandster.core.base_iteration import BaseIteration


class SuccessiveHalving(BaseIteration):

    def _advance_to_next_stage(self,
                               config_ids: List[Tuple[int, int, int]],
                               losses: np.ndarray) -> bool:
        """
            SuccessiveHalving simply continues the best based on the current loss.
        """
        ranks = np.argsort(np.argsort(losses))
        return ranks < self.num_configs[self.stage]
