import copy
import json
import os
from collections import ChainMap
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
                 meta_config: dict):
        self.meta_config = meta_config
        self.data: Dict[CandidateId, CandidateStructure] = dict(ChainMap(*data))

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
                inc = v.get_incumbent()
                if inc is not None:
                    tmp_list.append((inc.loss, k))
            except KeyError:
                pass

        if len(tmp_list) > 0:
            return min(tmp_list)[1]
        return None

    def get_runs_by_id(self, config_id: CandidateId) -> List[Result]:
        """
        returns a list of runs for a given config id
        """
        d = self.data[config_id]
        if d is None:
            return []
        return d.results

    def get_all_runs(self, ) -> List[Result]:
        """
        returns all runs performed
        :return:
        """
        all_runs = []
        for k in self.data.keys():
            all_runs.extend(self.get_runs_by_id(k))
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
            data[cs.cid] = cs

    with open(os.path.join(directory, 'results.json')) as fh:
        for line in fh:
            config_id, result, exception = json.loads(line)

            # TODO budget is not logged anymore
            budget = 0

            cid = CandidateId(*config_id).without_config()

            if result is not None:
                res = Result(result.get('status'), Configuration(data[cid].configspace, result.get('config')),
                             result.get('steps'), result.get('loss'), result.get('runtime'))
                data[cid].add_result(res)
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
