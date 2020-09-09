import copy
from collections import ChainMap
from typing import List, Dict, Optional, Tuple

from sklearn import clone

from dswizard.components.pipeline import FlexiblePipeline
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

    def get_incumbent(self) -> Optional[Tuple[FlexiblePipeline, CandidateStructure]]:
        """
        Find the incumbent.

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
            structure = self.data[min(tmp_list)[1]]
            # TODO pipeline is not fitted. Maybe store fitted pipeline?
            pipeline = clone(structure.pipeline)
            result = structure.get_incumbent()
            if result is None:
                raise ValueError('Incumbent structure has no config evaluations')
            pipeline.set_hyperparameters(result.config.get_dictionary())
            return pipeline, structure
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
