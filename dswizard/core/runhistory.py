import copy
import json
import os
from typing import List, Dict, Optional

from ConfigSpace import Configuration

from dswizard.core.model import CandidateId, CandidateStructure, Result


class RunHistory:
    """
    Object returned by the HB_master.run function

    This class offers a simple API to access the information from a Hyperband run.
    """

    def __init__(self,
                 data: List[Dict[CandidateId, CandidateStructure]],
                 algorithm_config: dict):
        self.HB_config = algorithm_config
        self.data = self._merge_results(data)

    @staticmethod
    def _merge_results(data: List[Dict[CandidateId, CandidateStructure]]) -> Dict[CandidateId, CandidateStructure]:
        """
        protected function to merge the list of results into one dictionary and 'normalize' the time stamps
        """
        new_dict = {}
        for it in data:
            new_dict.update(it)

        return new_dict

    def __getitem__(self, k):
        return self.data[k]

    def get_incumbent_id(self) -> Optional[CandidateId]:
        """
        Find the config_id of the incumbent.

        The incumbent here is the configuration with the smallest loss among all runs on the maximum budget! If no run
        finishes on the maximum budget, None is returned!
        """
        tmp_list = []
        for k, v in self.data.items():
            try:
                # only things run for the max budget are considered
                res = v.results[self.HB_config['max_budget']]
                if res is not None:
                    tmp_list.append((min([r.loss for r in res]), k))
            except KeyError:
                pass

        if len(tmp_list) > 0:
            return min(tmp_list)[1]
        return None

    def get_runs_by_id(self, config_id: CandidateId) -> List[Result]:
        """
        returns a list of runs for a given config id

        The runs are sorted by ascending budget, so '-1' will give the longest run for this config.
        """
        d = self.data[config_id]

        runs = []
        for budget in d.results.keys():
            for result in d.results.get(budget, []):
                runs.append(result)
        return runs

    def get_all_runs(self, only_largest_budget: bool = False) -> List[Result]:
        """
        returns all runs performed
        :param only_largest_budget: if True, only the largest budget for each configuration is returned. This makes
            sense if the runs are continued across budgets and the info field contains the information you care about.
            If False, all runs of a configuration are returned
        :return:
        """
        all_runs = []

        for k in self.data.keys():
            runs = self.get_runs_by_id(k)

            if len(runs) > 0:
                if only_largest_budget:
                    all_runs.append(runs[-1])
                else:
                    all_runs.extend(runs)

        return all_runs

    def get_id2config_mapping(self) -> Dict[CandidateId, CandidateStructure]:
        """
        returns a dict where the keys are the config_ids and the values are the actual configurations
        """
        return copy.deepcopy(self.data)

    def num_iterations(self) -> int:
        return max([k.iteration for k in self.data.keys()]) + 1


def logged_results_to_runhistory(directory: str) -> RunHistory:
    """
    function to import logged 'live-results' and return a RunHistory object

    You can load live run results with this function and the returned HB_result object gives you access to the results
    the same way a finished run would.

    :param directory: the directory containing the results.json and config.json files
    :return:
    """

    data = {}
    budget_set = set()

    with open(os.path.join(directory, 'structures.json')) as fh:
        for line in fh:
            raw = json.loads(line)
            cs = CandidateStructure.from_dict(raw)
            data[cs.id] = cs

    with open(os.path.join(directory, 'results.json')) as fh:
        for line in fh:
            config_id, budget, result, exception = json.loads(line)

            cid = CandidateId(*config_id).without_config()

            if result is not None:
                res = Result(result.get('status'), Configuration(data[cid].configspace, result.get('config')),
                             result.get('loss'), result.get('runtime'))
                data[cid].add_result(budget, res)
                budget_set.add(budget)

        # infer the hyperband configuration from the data
        budget_list = sorted(list(budget_set))
        algorithm_config = {
            'eta': None if len(budget_list) < 2 else budget_list[1] / budget_list[0],
            'min_budget': min(budget_set),
            'max_budget': max(budget_set),
            'budgets': budget_list,
            'max_SH_iter': len(budget_set),
        }
        return RunHistory([data], algorithm_config)
