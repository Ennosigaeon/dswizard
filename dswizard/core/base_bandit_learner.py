from __future__ import annotations

import abc
import logging
from typing import List, TYPE_CHECKING, Dict

if TYPE_CHECKING:
    from dswizard.core.base_iteration import BaseIteration
    from dswizard.core.model import CandidateStructure, Result


class BanditLearner(abc.ABC):

    def __init__(self, logger: logging.Logger = None):
        self.offset = 0
        self.meta_data = {}

        if logger is None:
            self.logger = logging.getLogger('Racing')
        else:
            self.logger = logger

        self.iterations: List[BaseIteration] = []
        self.max_iterations = 0

    @abc.abstractmethod
    def _get_next_iteration(self, iteration: int, iteration_kwargs: Dict) -> BaseIteration:
        """
        instantiates the next iteration

        Overwrite this to change the iterations for different optimizers
        :param iteration: the index of the iteration to be instantiated
        :param iteration_kwargs: additional kwargs for the iteration class. Defaults to empty dictionary
        :return: a valid HB iteration object
        """
        pass

    def next_candidate(self, iteration_kwargs: Dict = None) -> List[CandidateStructure]:
        """
        Returns the next CandidateStructure with an according budget.
        :param iteration_kwargs:
        :return:
        """
        n_iterations = self.max_iterations
        while True:
            next_candidate = None
            # find a new run to schedule
            for i in filter(lambda idx: not self.iterations[idx].is_finished, range(len(self.iterations))):
                next_candidate = self.iterations[i].get_next_candidate()
                if next_candidate is not None:
                    break

            if next_candidate is not None:
                # noinspection PyUnboundLocalVariable
                yield next_candidate
            else:
                # Ensure that current stage is completely done
                busy = any([not it.is_finished for it in self.iterations])
                if busy:
                    yield None
                    continue
                elif n_iterations > 0:  # we might be able to start the next iteration
                    iteration = len(self.iterations)
                    self.iterations.append(self._get_next_iteration(iteration, iteration_kwargs))
                    n_iterations -= 1
                else:
                    # Done
                    break

    def reset(self, offset: int):
        self.offset = offset
        self.iterations = []

    def register_result(self, cs: CandidateStructure, result: Result) -> CandidateStructure:
        return self.iterations[-1].register_result(cs, result)
