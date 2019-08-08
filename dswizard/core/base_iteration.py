import abc
import logging
from typing import List, Optional, Dict

import math
import numpy as np

from dswizard.core.base_structure_generator import BaseStructureGenerator
from dswizard.core.model import CandidateId, CandidateStructure
from dswizard.core.runhistory import JsonResultLogger


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
                 timeout: float = None,
                 sampler: BaseStructureGenerator = None,
                 logger: logging.Logger = None,
                 result_logger: JsonResultLogger = None):
        """

        :param iteration: The current Hyperband iteration index.
        :param num_candidates: the number of configurations in each stage of SH
        :param budgets: the budget associated with each stage
        :param timeout: the maximum timeout for evaluating a single configuration
        :param sampler: a function that returns a valid configuration. Its only argument should be the budget
            that this config is first scheduled for. This might be used to pick configurations that perform best after
            this particular budget is exhausted to build a better autoML system.
        :param logger: a logger
        :param result_logger: a result logger that writes live results to disk
        """

        self.data: Dict[CandidateId, CandidateStructure] = {}  # this holds all the candidates of this iteration
        self.is_finished = False
        self.iteration = iteration
        self.stage = 0  # internal iteration, but different name for clarity
        self.budgets = budgets
        self.timeout = timeout
        self.num_candidates = num_candidates
        self.actual_num_candidates = [0] * len(num_candidates)
        self.sampler = sampler
        self.num_running = 0
        if logger is None:
            self.logger = logging.getLogger('Iteration')
        else:
            self.logger = logger
        self.result_logger = result_logger

    def add_candidate(self, candidate: CandidateStructure = None) -> CandidateId:
        """
        function to add a new configuration to the current iteration
        :param candidate: The configuration to add. If None, a configuration is sampled from the config_sampler
        :return: The id of the new configuration
        """
        if candidate is None:
            candidate = self.sampler.get_candidate(self.budgets[self.stage])

        if self.is_finished:
            raise RuntimeError("This iteration is finished, you can't add more configurations!")

        if self.actual_num_candidates[self.stage] == self.num_candidates[self.stage]:
            raise RuntimeError("Can't add another candidate to stage {} in iteration {}.".format(self.stage,
                                                                                                 self.iteration))

        candidate_id = CandidateId(self.iteration, self.actual_num_candidates[self.stage])
        candidate.id = candidate_id
        timeout = math.ceil(self.budgets[self.stage] * self.timeout) if self.timeout is not None else None
        candidate.timeout = timeout

        self.data[candidate_id] = candidate
        self.actual_num_candidates[self.stage] += 1

        if self.result_logger is not None:
            self.result_logger.new_structure(candidate)

        return candidate_id

    def register_result(self, cs: CandidateStructure) -> None:
        """
        function to register the result of a job

        This function is called from HB_master, don't call this from your script.
        :param cs: Finished CandidateStructure
        :return:
        """

        if self.is_finished:
            raise RuntimeError("This HB iteration is finished, you can't register more results!")
        cs.status = 'REVIEW'
        self.num_running -= 1

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

        for candidate in self.data.values():
            if candidate.status == 'QUEUED':
                assert candidate.budget == self.budgets[self.stage], \
                    'Configuration budget does not align with current stage!'
                candidate.status = 'RUNNING'
                self.num_running += 1
                return candidate

        # check if there are still slots to fill in the current stage and return that
        if self.actual_num_candidates[self.stage] < self.num_candidates[self.stage]:
            self.add_candidate()
            return self.get_next_candidate()

        if self.num_running == 0:
            # at this point a stage is completed
            self.logger.debug('Stage {} completed'.format(self.stage))
            self._process_results()
            return self.get_next_candidate()

        return None

    @abc.abstractmethod
    def _advance_to_next_stage(self, losses: np.ndarray) -> np.ndarray:
        """
        Function that implements the strategy to advance configs within this iteration

        Overload this to implement different strategies, like SuccessiveHalving, SuccessiveResampling.
        :param losses: losses of the run on the current budget
        :return: A boolean for each entry in config_ids indicating whether to advance it or not
        """
        raise NotImplementedError('_advance_to_next_stage not implemented for {}'.format(type(self).__name__))

    def _process_results(self) -> None:
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
            self.finish_up()
            return

        budgets = [self.data[cid].budget for cid in candidate_ids]
        if len(set(budgets)) > 1:
            raise RuntimeError('Not all configurations have the same budget!')
        budget = self.budgets[self.stage - 1]

        losses = np.array([self.data[cid].get_incumbent(budget).loss for cid in candidate_ids])
        advance = self._advance_to_next_stage(losses)

        for i, cid in enumerate(candidate_ids):
            if advance[i]:
                self.logger.debug('Advancing candidate {} to next budget {}'.format(cid, self.budgets[self.stage]))

                candidate = self.data[cid]
                candidate.status = 'QUEUED'
                candidate.budget = self.budgets[self.stage]
                candidate.timeout = math.ceil(candidate.budget * self.timeout) if self.timeout is not None else None
                self.actual_num_candidates[self.stage] += 1
            else:
                self.data[cid].status = 'TERMINATED'

    def finish_up(self) -> None:
        self.is_finished = True

        for k, v in self.data.items():
            assert v.status in ['TERMINATED', 'REVIEW', 'CRASHED'], 'Configuration has not finshed yet!'
            v.status = 'COMPLETED'
