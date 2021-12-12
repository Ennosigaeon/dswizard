import os
import time
from typing import List, Dict, Optional, Tuple, Any

from ConfigSpace import ConfigurationSpace
from ConfigSpace.read_and_write import json as config_json
from sklearn import clone

from dswizard.core.model import CandidateId, CandidateStructure, Result, StatusType, MetaInformation
from dswizard.pipeline.pipeline import FlexiblePipeline
from dswizard.util.util import model_file, metric_sign


class RunHistory:
    """
    Object returned by the HB_master.run function

    This class offers a simple API to access the information from a Hyperband run.
    """

    def __init__(self,
                 data: Dict[CandidateId, CandidateStructure],
                 meta_information: MetaInformation,
                 structure_xai: Dict[str, Any],
                 default_configspace: Optional[ConfigurationSpace] = None):
        # Map structures to JSON structure
        structures = []
        for s in data.values():
            structure = s.as_dict()
            # Store budget in each configuration instead of only in structure. Only necessary for compatability with
            # other AutoML frameworks
            structure['configs'] = [r.as_dict(s.budget, loss_sign=metric_sign(meta_information.metric))
                                    for r in s.results]
            del structure['budget']
            del structure['cfg_keys']

            structures.append(structure)

        self.data = data
        self.meta_information = meta_information
        self.complete_data = {
            'meta': meta_information.as_dict(),
            'default_configspace': config_json.write(default_configspace) if default_configspace is not None else None,
            'structures': structures,
            'explanations': {
                'structures': structure_xai
            }
        }

    @staticmethod
    def create(data: Dict[CandidateId, CandidateStructure],
               meta_information: MetaInformation,
               iterations: Dict,
               workdir: str,
               structure_xai: Dict[str, Any]
               ):
        # Collapse data to merge identical structures
        reverse: Dict[CandidateStructure, CandidateId] = {}
        collapsed_data: Dict[CandidateId, CandidateStructure] = {}
        for cid, structure in data.items():
            if structure not in reverse:
                reverse[structure] = cid
                collapsed_data[cid] = structure
            else:
                new_structure = collapsed_data[reverse[structure]]
                offset = len(new_structure.results)

                for idx, res in enumerate(structure.results):
                    # Change config id and add to new structure
                    res.cid = new_structure.cid.with_config(offset + idx)
                    new_structure.add_result(res)

                    if res.status == StatusType.SUCCESS:
                        # Rename model files so that they can be found during ensemble construction
                        os.rename(os.path.join(workdir, model_file(cid.with_config(idx))),
                                  os.path.join(workdir, model_file(res.cid)))

        for s in data.values():
            for res in s.results:
                res.model_file = os.path.abspath(os.path.join(workdir, model_file(res.cid)))

        # Fill in missing meta-information
        meta_information.end_time = time.time()
        meta_information.n_configs = sum([len(c.results) for c in data.values()])
        meta_information.n_structures = len(data)
        meta_information.iterations = iterations
        meta_information.incumbent = min(
            [s.get_incumbent().loss for s in collapsed_data.values() if s.get_incumbent() is not None]
        )

        return RunHistory(data, meta_information, structure_xai)

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

    def _repr_mimebundle_(self, include, exclude):
        return {
            'application/xautoml+json': self.complete_data
        }

    def explain(self):
        try:
            # noinspection PyPackageRequirements
            from IPython.core.display import display
            # noinspection PyTypeChecker
            return display(self)
        except ImportError:
            return str(self)
