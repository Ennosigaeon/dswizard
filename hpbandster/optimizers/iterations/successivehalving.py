from typing import List, Tuple

import numpy as np

from hpbandster.core import BaseIteration


class SuccessiveHalving(BaseIteration):

    def _advance_to_next_stage(self,
                               config_ids: List[Tuple[int, int, int]],
                               losses: np.ndarray) -> bool:
        """
            SuccessiveHalving simply continues the best based on the current loss.
        """
        ranks = np.argsort(np.argsort(losses))
        return ranks < self.num_configs[self.stage]
