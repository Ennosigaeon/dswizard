import copy
import json
import os
from typing import List, Dict, Optional

from ConfigSpace import Configuration

from dswizard.core.model import CandidateId, Job, CandidateStructure, Result


class JsonResultLogger:
    def __init__(self, directory: str, overwrite: bool = False):
        """
        convenience logger for 'semi-live-results'

        Logger that writes job results into two files (configs.json and results.json). Both files contain proper json
        objects in each line.  This version opens and closes the files for each result. This might be very slow if
        individual runs are fast and the filesystem is rather slow (e.g. a NFS).
        :param directory: the directory where the two files 'configs.json' and 'results.json' are stored
        :param overwrite: In case the files already exist, this flag controls the
            behavior:
                * True:   The existing files will be overwritten. Potential risk of deleting previous results
                * False:  A FileExistsError is raised and the files are not modified.
        """

        os.makedirs(directory, exist_ok=True)

        self.structure_fn = os.path.join(directory, 'structures.json')
        self.results_fn = os.path.join(directory, 'results.json')

        try:
            with open(self.structure_fn, 'x'):
                pass
        except FileExistsError:
            if overwrite:
                with open(self.structure_fn, 'w'):
                    pass
            else:
                raise FileExistsError('The file {} already exists.'.format(self.structure_fn))

        try:
            with open(self.results_fn, 'x'):
                pass
        except FileExistsError:
            if overwrite:
                with open(self.results_fn, 'w'):
                    pass
            else:
                raise FileExistsError('The file {} already exists.'.format(self.structure_fn))

        self.structure_ids = set()

    def new_structure(self, structure: CandidateStructure) -> None:
        if structure.id not in self.structure_ids:
            self.structure_ids.add(structure.id)
            with open(self.structure_fn, 'a') as fh:
                fh.write(json.dumps(structure.as_dict()))
                fh.write('\n')

    def log_evaluated_config(self, job: Job) -> None:
        if job.id.without_config() not in self.structure_ids:
            # should never happen! TODO: log warning here!
            self.structure_ids.add(job.id)
            with open(self.structure_fn, 'a') as fh:
                fh.write(json.dumps([job.id.as_tuple(), job.config, {}]))
                fh.write('\n')
        with open(self.results_fn, 'a') as fh:
            fh.write(
                json.dumps([job.id.as_tuple(), job.budget, job.result.as_dict() if job.result is not None else None])
            )
            fh.write("\n")


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

    def _merge_results(self, data: List[Dict[CandidateId, CandidateStructure]]) -> Dict[
        CandidateId, CandidateStructure]:
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

            id = CandidateId(*config_id).without_config()

            if result is not None:
                res = Result(result.get('status'), Configuration(data[id].configspace, result.get('config')),
                             result.get('loss'), result.get('runtime'))
                data[id].add_result(budget, res)
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
