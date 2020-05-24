from __future__ import annotations

import logging
from typing import List, TYPE_CHECKING

import numpy as np

from dswizard.core.base_iteration import BaseIteration

if TYPE_CHECKING:
    from dswizard.core.logger import JsonResultLogger
    from dswizard.core.base_structure_generator import BaseStructureGenerator


class SuccessiveResampling(BaseIteration):

    def __init__(self,
                 iteration: int,
                 num_candidates: List[int],
                 budgets: List[float],
                 structure_generator: BaseStructureGenerator = None,
                 logger: logging.Logger = None,
                 result_logger: JsonResultLogger = None,
                 resampling_rate=0.5,
                 min_samples_advance=1):
        """
        Iteration class to resample new configurations along side keeping the good ones in SuccessiveHalving.
        :param iteration:
        :param num_candidates:
        :param budgets:
        :param structure_generator:
        :param resampling_rate: fraction of configurations that are resampled at each stage
        :param min_samples_advance: number of samples that are guaranteed to proceed to the next stage regardless of
            the fraction.
        :param kwargs:
        """

        super().__init__(iteration, num_candidates, budgets, structure_generator, logger, result_logger)
        self.resampling_rate = resampling_rate
        self.min_samples_advance = min_samples_advance

    def _advance_to_next_stage(self, losses) -> np.ndarray:
        """
        SuccessiveHalving simply continues the best based on the current loss.
        """

        ranks = np.argsort(np.argsort(losses))
        return ranks < max(self.min_samples_advance, self.num_candidates[self.stage] * (1 - self.resampling_rate))
