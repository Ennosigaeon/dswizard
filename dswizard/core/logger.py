from __future__ import annotations

import json
import logging
import os
import pickle
import shutil
from typing import List, Tuple, Dict

import joblib
import networkx as nx
from ConfigSpace import Configuration
from sklearn.ensemble import VotingClassifier
from slugify import slugify

from dswizard.core.constants import MODEL_DIR
from dswizard.core.model import CandidateStructure, CandidateId, Result, StatusType
from dswizard.core.model import PartialConfig
from dswizard.core.runhistory import RunHistory
from dswizard.pipeline.pipeline import FlexiblePipeline
from dswizard.util import util


class ResultLogger:
    def __init__(self, directory: str, tmp_dir: str):
        # Remove old results
        try:
            shutil.rmtree(directory)
        except FileNotFoundError:
            pass

        os.makedirs(directory, exist_ok=True)
        os.makedirs(os.path.join(directory, MODEL_DIR), exist_ok=True)

        self.directory = directory
        self.tmp_dir = tmp_dir
        self.structure_fn = os.path.join(directory, 'structures.json')
        self.results_fn = os.path.join(directory, 'results.json')
        self.structure_ids = set()

    def new_structure(self, structure: CandidateStructure, draw_structure: bool = False) -> None:
        if structure.cid.without_config() not in self.structure_ids:
            self.structure_ids.add(structure.cid.without_config())
            with open(self.structure_fn, 'a') as fh:
                fh.write(json.dumps(structure.as_dict()))
                fh.write('\n')

            # Results may already be created during structure creation
            for idx, result in enumerate(structure.results):
                result.cid = result.cid.with_config(idx)
                self.log_evaluated_config(structure, result)

            if draw_structure:
                g = structure.pipeline.to_networkx()
                h = nx.nx_agraph.to_agraph(g)
                h.draw(f'{self.directory}/{structure.cid}.png', prog='dot')

    def log_evaluated_config(self, structure: CandidateStructure, result: Result) -> None:
        cid = result.cid
        if cid.without_config() not in self.structure_ids:
            # should never happen!
            raise ValueError(f'Unknown structure {cid.without_config()}')
        with open(self.results_fn, 'a') as fh:
            fh.write(
                json.dumps([cid.as_tuple(), result.as_dict() if result is not None else None])
            )
            fh.write("\n")

        if result.status == StatusType.SUCCESS:
            self._store_fitted_model(structure, cid)

    def _store_fitted_model(self, structure: CandidateStructure, cid: CandidateId) -> None:
        try:
            file = os.path.join(self.tmp_dir, util.model_file(cid))
            with open(file, 'rb') as f:
                pipeline = joblib.load(f)[0]
        except FileNotFoundError:
            # Load partial models with default hyperparameters
            steps = []
            for name, _ in structure.steps:
                with open(os.path.join(self.tmp_dir, f'step_{slugify(name)}.pkl'), 'rb') as f:
                    model = joblib.load(f)[0]
                    steps.append((name, model))

            pipeline = FlexiblePipeline(steps=steps)
            config = pipeline.get_hyperparameter_search_space().get_default_configuration()
            config.origin = 'Default'
            pipeline.configuration = config

        file = os.path.join(self.directory, MODEL_DIR, util.model_file(cid))
        with open(file, 'wb') as f:
            joblib.dump(pipeline, f)

    def log_run_history(self, runhistory: RunHistory, suffix: str = 'None') -> None:
        with open(os.path.join(self.directory, f'runhistory_{suffix}.json'), 'w') as fh:
            fh.write(json.dumps(runhistory.complete_data))
        with open(os.path.join(self.directory, f'runhistory_{suffix}.pkl'), 'wb') as fh:
            pickle.dump(runhistory, fh)

    def log_ensemble(self, ensemble: VotingClassifier, suffix: str = 'None') -> None:
        with open(os.path.join(self.directory, f'ensemble_{suffix}.pkl'), 'wb') as fh:
            pickle.dump(ensemble, fh)

    def load(self) -> Dict[CandidateId, CandidateStructure]:
        structures = {}
        with open(self.structure_fn, 'r') as structure_file:
            for line in structure_file:
                raw = json.loads(line)
                cs = CandidateStructure.from_dict(raw)
                structures[cs.cid] = cs

        with open(self.results_fn, 'r') as result_file:
            for line in result_file:
                raw = json.loads(line)
                cid = CandidateId(*raw[0])

                cs = structures[cid.without_config()]
                res = Result.from_dict(raw[1], cs.configspace)
                cs.add_result(res)
        return structures


class ProcessLogger:

    def __init__(self, directory: str, config_id, logger: logging.Logger = None):
        os.makedirs(directory, exist_ok=True)
        self.prefix = os.path.join(directory, '{}-{}-{}'.format(*config_id.as_tuple()))
        self.partial_configs: List[PartialConfig] = []

        if logger is None:
            self.logger = logging.getLogger('ProcessLogger')
        else:
            self.logger = logger

        self.file = f'{self.prefix}.json'
        with open(self.file, 'w'):
            pass

    def new_step(self, name: str, config: PartialConfig) -> None:
        self.partial_configs.append(config)
        with open(self.file, 'a') as fh:
            fh.write(json.dumps([name, config.as_dict(), config.config.origin]))
            fh.write('\n')

    def get_config(self, pipeline: FlexiblePipeline) -> Configuration:
        return self._merge_configs(self.partial_configs, pipeline)

    def restore_config(self, pipeline: FlexiblePipeline) -> Tuple[Configuration, List[PartialConfig]]:
        partial_configs: List[PartialConfig] = []

        with open(self.file) as fh:
            for line in fh:
                name, partial_config, origin = json.loads(line)
                partial_config = PartialConfig.from_dict(partial_config, origin)

                partial_configs.append(partial_config)

        partial_configs = self._add_missing_configs(partial_configs, pipeline)
        config = self._merge_configs(partial_configs, pipeline)
        os.remove(self.file)
        return config, partial_configs

    def _add_missing_configs(self, partial_configs: List[PartialConfig], pipeline: FlexiblePipeline) -> \
            List[PartialConfig]:
        if len(partial_configs) == 0:
            self.logger.warning('Encountered job without any partial configurations. Simulating complete config')

        missing_steps = set(pipeline.all_names())
        for partial_config in partial_configs:
            missing_steps.remove(partial_config.name)

        # Create random configuration for missing steps
        latest_mf = None if len(partial_configs) == 0 else partial_configs[-1].mf
        for name in missing_steps:
            config = pipeline.get_step(name).get_hyperparameter_search_space(mf=latest_mf) \
                .sample_configuration()
            config.origin = 'Random Search'
            # noinspection PyTypeChecker
            partial_configs.append(PartialConfig(None, config, name, latest_mf))
        return partial_configs

    def _merge_configs(self, partial_configs: List[PartialConfig], pipeline: FlexiblePipeline) -> Configuration:
        try:
            return util.merge_configurations(partial_configs, pipeline.configuration_space)
        except ValueError as ex:
            self.logger.error('Failed to reconstruct global config.\n'
                              f'Exception: {ex}\nConfigSpace: {pipeline.configuration_space}')
            raise ex
