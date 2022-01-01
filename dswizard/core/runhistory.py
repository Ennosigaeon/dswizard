import logging
import os
import time
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
from sklearn import clone

from dswizard.components.util import prefixed_name
from dswizard.core.model import CandidateId, CandidateStructure, ConfigKey, Result, StatusType, MetaInformation
from dswizard.pipeline.pipeline import FlexiblePipeline
from dswizard.util.util import model_file, metric_sign, merge_configurations


class RunHistory:
    """
    Object returned by the HB_master.run function

    This class offers a simple API to access the information from a Hyperband run.
    """

    def __init__(self,
                 data: Dict[CandidateId, CandidateStructure],
                 meta_information: MetaInformation,
                 structure_xai: Dict[str, Any],
                 config_xai: Dict[ConfigKey, Any]):
        # Map structures to JSON structure
        structures = []

        config_explanations = {}

        for s in data.values():
            structure = s.as_dict()
            # Store budget in each configuration instead of only in structure. Only necessary for compatability with
            # other AutoML frameworks
            structure['configs'] = [r.as_dict(s.budget, loss_sign=metric_sign(meta_information.metric))
                                    for r in s.results]
            structure['config_explanations'] = {}

            # Extract and merge all (partial-) configurations
            explanations_for_steps = [config_xai[key] for key in s.cfg_keys]
            cids = set.intersection(*[set(e.keys()) for e in explanations_for_steps])
            for cid in cids:
                try:
                    partial_configs = []
                    loss = []
                    marginalization = {}
                    for explanations_for_step in explanations_for_steps:
                        pcs = explanations_for_step[cid]['candidates']
                        prefix = pcs[0].name if len(pcs) > 0 else None
                        partial_configs.append(pcs)
                        loss.append(explanations_for_step[cid]['loss'])
                        marginalization = {**marginalization, **{
                            prefixed_name(prefix, key): value for key, value in
                            explanations_for_step[cid]['marginalization'].items()
                        }}

                    loss = np.array(loss).T.mean(axis=1)
                    configs = [merge_configurations(pc.tolist(), s.configspace).get_dictionary()
                               for pc in np.array(partial_configs).T]

                    config_explanations[cid] = {
                        'loss': np.clip(loss, -1000, 100).tolist(),
                        'candidates': configs,
                        'marginalization': marginalization
                    }
                except ValueError as ex:
                    logging.error('Failed to reconstruct global config.\n'
                                  f'Exception: {ex}\nConfigSpace: {structure}')

            del structure['budget']
            del structure['cfg_keys']

            structures.append(structure)

        self.data = data
        self.meta_information = meta_information
        self.complete_data = {
            'meta': meta_information.as_dict(),
            'structures': structures,
            'explanations': {
                'structures': structure_xai,
                'configs': config_explanations
            }
        }

    @staticmethod
    def create(data: Dict[CandidateId, CandidateStructure],
               meta_information: MetaInformation,
               iterations: Dict,
               workdir: str,
               structure_xai: Dict[str, Any],
               config_xai: Dict[ConfigKey, Any]
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
                    old_cid = res.cid
                    res.cid = new_structure.cid.with_config(offset + idx)
                    new_structure.add_result(res)

                    # Rename config_xai entries
                    for key in structure.cfg_keys:
                        if key in config_xai and old_cid.external_name in config_xai[key]:
                            config_xai[key][res.cid.external_name] = config_xai[key][old_cid.external_name]
                            del config_xai[key][old_cid.external_name]

                    if res.status == StatusType.SUCCESS:
                        # Rename model files so that they can be found during ensemble construction
                        os.rename(os.path.join(workdir, model_file(cid.with_config(idx))),
                                  os.path.join(workdir, model_file(res.cid)))

        for s in data.values():
            for res in s.results:
                res.model_file = os.path.abspath(os.path.join(workdir, model_file(res.cid)))

        # Fill in missing meta-information
        meta_information.end_time = time.time()
        meta_information.n_configs = sum([len(c.results) for c in collapsed_data.values()])
        meta_information.n_structures = len(collapsed_data)
        meta_information.iterations = iterations
        meta_information.incumbent = min(
            [s.get_incumbent().loss for s in collapsed_data.values() if s.get_incumbent() is not None]
        )

        return RunHistory(collapsed_data, meta_information, structure_xai, config_xai)

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
