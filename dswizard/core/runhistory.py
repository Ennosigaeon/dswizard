import os
from typing import List, Dict, Optional, Tuple

from sklearn import clone

from dswizard.core.model import CandidateId, CandidateStructure, Result, StatusType
from dswizard.pipeline.pipeline import FlexiblePipeline
from dswizard.util.util import model_file


class RunHistory:
    """
    Object returned by the HB_master.run function

    This class offers a simple API to access the information from a Hyperband run.
    """

    def __init__(self,
                 data: Dict[CandidateId, CandidateStructure],
                 meta_config: dict,
                 workdir: str):
        self.meta_config = meta_config

        # Collapse data to merge identical structures
        reverse = {}
        self.data: Dict[CandidateId, CandidateStructure] = {}
        for cid, structure in data.items():
            if structure not in reverse:
                reverse[structure] = cid
                self.data[cid] = structure
            else:
                old_cid = reverse[structure]
                prev = self.data[old_cid]
                offset = len(prev.results)

                for i, r in enumerate(structure.results):
                    prev.results.append(r)
                    if r.status == StatusType.SUCCESS:
                        # Rename model files so that they can be found during ensemble construction
                        os.rename(os.path.join(workdir, model_file(cid.with_config(i))),
                                  os.path.join(workdir, model_file(old_cid.with_config(offset + i))))

    def __getitem__(self, k: CandidateId) -> CandidateStructure:
        return self.data[k]

    def get_incumbent(self) -> Tuple[Optional[FlexiblePipeline], Optional[CandidateStructure]]:
        """
        Find the incumbent.

        The incumbent here is the configuration with the smallest loss among all runs on the maximum budget! If no run
        finishes on the maximum budget, None is returned!
        """
        tmp_list = []
        for k, v in self.data.items():
            try:
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
        return None, None

    def get_all_runs(self, ) -> List[Tuple[CandidateId, Result]]:
        """
        returns all runs performed
        :return:
        """
        all_runs = []
        for structure in self.data.values():
            all_runs.extend([(structure.cid.with_config(idx), res) for idx, res in enumerate(structure.results)])
        return all_runs

    def get_all_pipelines(self) -> List[Tuple[FlexiblePipeline, Result]]:
        """
        returns all successful pipelines
        :return:
        """
        all_pipelines = []
        for structure in self.data.values():
            for result in structure.results:
                if result.status == StatusType.SUCCESS:
                    pipeline = clone(structure.pipeline)
                    pipeline.set_hyperparameters(result.config.get_dictionary())
                    all_pipelines.append(pipeline)
        return all_pipelines
