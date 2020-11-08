from typing import Optional

import math
import numpy as np

from dswizard.core.base_iteration import BaseIteration
from dswizard.core.model import CandidateStructure


class PseudoIteration(BaseIteration):

    def __init__(self, iteration: int, budget: int = 2):
        # noinspection PyTypeChecker
        super().__init__(iteration, [math.inf], [budget])

    def get_next_candidate(self) -> Optional[CandidateStructure]:
        return self._add_candidate()

    def _advance_to_next_stage(self, losses: np.ndarray) -> np.ndarray:
        pass
