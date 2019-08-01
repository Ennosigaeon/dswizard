from typing import List, Callable, Tuple

import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace

from hpbandster.core.base_iteration import BaseIteration


class SuccessiveResampling(BaseIteration):

    def __init__(self, HPB_iter: int, num_configs: List[int], budgets: List[float],
                 config_sampler: Callable[[float], Tuple[ConfigurationSpace, dict]],
                 *args, resampling_rate=0.5, min_samples_advance=1, **kwargs):
        """
        Iteration class to resample new configurations along side keeping the good ones in SuccessiveHalving.
        :param HPB_iter:
        :param num_configs:
        :param budgets:
        :param config_sampler:
        :param args:
        :param resampling_rate: fraction of configurations that are resampled at each stage
        :param min_samples_advance: number of samples that are guaranteed to proceed to the next stage regardless of
            the fraction.
        :param kwargs:
        """

        super().__init__(HPB_iter, num_configs, budgets, config_sampler)
        self.resampling_rate = resampling_rate
        self.min_samples_advance = min_samples_advance

    def _advance_to_next_stage(self, losses):
        """
        SuccessiveHalving simply continues the best based on the current loss.
        """

        ranks = np.argsort(np.argsort(losses))
        return ranks < max(self.min_samples_advance, self.num_configs[self.stage] * (1 - self.resampling_rate))
