from __future__ import annotations

import abc
import logging
from typing import List, Optional, Dict

import numpy as np

from dswizard.core.model import CandidateId, CandidateStructure, Result


class BaseIteration(abc.ABC):
    """
    Base class for various iteration possibilities. This decides what configuration should be run on what budget next.
    Typical choices are e.g. successive halving. Results from runs are processed and (depending on the implementations)
    determine the further development.
    """

    def __init__(self,
                 iteration: int,
                 num_candidates: List[int],
                 budgets: List[float],
                 logger: logging.Logger = None):
        """

        :param iteration: The current Hyperband repetition index.
        :param num_candidates: the number of configurations in each stage of SH
        :param budgets: the budget associated with each stage
        :param logger: a logger
        """

        self.data: Dict[CandidateId, CandidateStructure] = {}  # this holds all the candidates of this iteration
        self.is_finished = False
        self.iteration = iteration
        self.stage = 0  # internal iteration, but different name for clarity
        self.budgets = budgets
        self.num_candidates = num_candidates
        self.actual_num_candidates = [0] * len(num_candidates)
        self.num_running = 0
        if logger is None:
            self.logger = logging.getLogger('Bandit')
        else:
            self.logger = logger

    def register_result(self, cs: CandidateStructure, result: Result) -> CandidateStructure:
        """
        function to register the result of a job

        :param cs: Finished CandidateStructure
        :param result:
        :return:
        """

        if self.is_finished:
            raise RuntimeError("This SuccessiveHalving iteration is finished, you can't register more results!")
        cs = self.data[cs.cid]
        cs.results.append(result)
        cs.status = 'REVIEW'
        self.num_running -= 1
        return cs

    def replace_proxy(self, cs: CandidateStructure):
        assert self.data[cs.cid].is_proxy()
        self.data[cs.cid] = cs

    def get_next_candidate(self) -> Optional[CandidateStructure]:
        """
        function to return the next configuration and budget to run.

        This function is called from HB_master, don't call this from your script. It returns None if this run of SH is
        finished or there are pending jobs that need to finish to progress to the next stage.

        If there are empty slots to be filled in the current SH stage (which never happens in the original SH version),
        a new configuration will be sampled and scheduled to run next.
        :return: Tuple with ConfigId and Datum
        """

        if self.is_finished:
            return None

        # Check if candidates exists from previous stage
        candidates = list(filter(lambda cid: self.data[cid].status == 'QUEUED', self.data.keys()))
        for cid in candidates:
            candidate = self.data[cid]
            assert candidate.budget == self.budgets[self.stage], 'Config budget does not align with current stage!'
            candidate.status = 'RUNNING'
            self.num_running += int(candidate.budget)
            return candidate

        # check if there are still slots to fill in the current stage and return that
        if self.actual_num_candidates[self.stage] < self.num_candidates[self.stage]:
            candidate = self._add_candidate()
            candidate.status = 'RUNNING'
            self.num_running += int(candidate.budget)
            return candidate
        elif self.num_running == 0:
            # at this point a stage is completed
            self.logger.info('Stage {} completed'.format(self.stage))
            self._finish_stage()
            return self.get_next_candidate()
        else:
            return None

    def _add_candidate(self) -> CandidateStructure:
        """
        function to add a new configuration to the current iteration
        :return: The id of the new configuration
        """
        if self.is_finished:
            raise RuntimeError("This iteration is finished, you can't add more configurations!")

        if self.actual_num_candidates[self.stage] == self.num_candidates[self.stage]:
            raise RuntimeError("Can't add another candidate to stage {}.".format(self.stage))

        candidate = CandidateStructure.proxy()
        candidate.budget = self.budgets[self.stage]

        candidate_id = CandidateId(self.iteration, self.actual_num_candidates[self.stage])
        candidate.cid = candidate_id
        # timeout = math.ceil(self.budgets[self.stage] * self.timeout) if self.timeout is not None else None
        # candidate.timeout = timeout

        self.data[candidate_id] = candidate
        self.actual_num_candidates[self.stage] += 1

        return candidate

    def _finish_stage(self) -> None:
        """
        function that is called when a stage is completed and needs to be analyzed before further computations.

        The code here implements the original SH algorithms by advancing the k-best (lowest loss) configurations at the
        current budget. k is defined by the num_configs list (see __init__) and the current stage value.

        For more advanced methods like resampling after each stage, overload this function only.
        """
        self.stage += 1

        # collect all candidate_ids that need to be compared
        candidate_ids = list(filter(lambda cid: self.data[cid].status == 'REVIEW', self.data.keys()))

        if self.stage >= len(self.num_candidates):
            self._finish_up()
            return

        budgets = [self.data[cid].budget for cid in candidate_ids]
        if len(set(budgets)) > 1:
            raise RuntimeError('Not all configurations have the same budget!')

        losses = np.array([self.data[cid].get_incumbent().loss for cid in candidate_ids])
        advance = self._advance_to_next_stage(losses)

        for i, cid in enumerate(candidate_ids):
            if advance[i]:
                self.logger.debug('Advancing candidate structure {} to next budget {} with loss {}'
                                  .format(cid, self.budgets[self.stage], losses[i]))

                candidate = self.data[cid]
                candidate.status = 'QUEUED'
                candidate.budget = self.budgets[self.stage]
                # candidate.timeout = math.ceil(candidate.budget * self.timeout) if self.timeout is not None else None
                self.actual_num_candidates[self.stage] += 1
            else:
                self.data[cid].status = 'TERMINATED'

    def _finish_up(self) -> None:
        self.is_finished = True

        for k, v in self.data.items():
            assert v.status in ['TERMINATED', 'REVIEW', 'CRASHED'], 'Configuration has not finshed yet!'
            v.status = 'COMPLETED'

    @abc.abstractmethod
    def _advance_to_next_stage(self, losses: np.ndarray) -> np.ndarray:
        """
        Function that implements the strategy to advance configs within this iteration

        Overload this to implement different strategies, like SuccessiveHalving, SuccessiveResampling.
        :param losses: losses of the run on the current budget
        :return: A boolean for each entry in config_ids indicating whether to advance it or not
        """
        raise NotImplementedError('_advance_to_next_stage not implemented for {}'.format(type(self).__name__))
