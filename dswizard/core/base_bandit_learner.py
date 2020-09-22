from __future__ import annotations

import abc
import logging
from typing import List, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from dswizard.core.base_iteration import BaseIteration
    from dswizard.core.base_structure_generator import BaseStructureGenerator
    from dswizard.core.model import CandidateStructure, Job, Dataset


class BanditLearner(abc.ABC):

    def __init__(self,
                 structure_generator: BaseStructureGenerator = None,
                 logger: logging.Logger = None):
        self.offset = 0
        self.structure_generator = structure_generator
        self.meta_data = {}

        if logger is None:
            self.logger = logging.getLogger('Racing')
        else:
            self.logger = logger

        self.iterations: List[BaseIteration] = []
        self.max_iterations = 0

    @abc.abstractmethod
    def _get_next_iteration(self, iteration: int, iteration_kwargs: dict) -> BaseIteration:
        """
        instantiates the next iteration

        Overwrite this to change the iterations for different optimizers
        :param iteration: the index of the iteration to be instantiated
        :param iteration_kwargs: additional kwargs for the iteration class. Defaults to empty dictionary
        :return: a valid HB iteration object
        """
        pass

    def next_candidate(self, ds: Dataset, iteration_kwargs: dict = None) -> List[Tuple[CandidateStructure, int]]:
        """
        Returns the next CandidateStructure with an according budget.
        :param ds:
        :param iteration_kwargs:
        :return:
        """
        n_iterations = self.max_iterations
        while True:
            next_candidate = None
            # find a new run to schedule
            for i in filter(lambda idx: not self.iterations[idx].is_finished, range(len(self.iterations))):
                next_candidate = self.iterations[i].get_next_candidate(ds)
                if next_candidate is not None:
                    break

            if next_candidate is not None:
                # noinspection PyUnboundLocalVariable
                yield next_candidate, i
            else:
                # TODO if multiple workers, check that really all workers have finished before starting next iteration
                if n_iterations > 0:  # we might be able to start the next iteration
                    iteration = len(self.iterations)
                    self.logger.info('Starting iteration {}'.format(iteration))
                    self.iterations.append(self._get_next_iteration(iteration, iteration_kwargs))
                    n_iterations -= 1
                else:
                    # Done
                    break

    def reset(self, offset: int):
        self.offset = offset
        self.iterations = []

    def register_result(self, job: Job, update_model: bool = True):
        self.iterations[-1].register_result(job.cs)
        self.structure_generator.register_result(job.cs, job.result, update_model=update_model)
