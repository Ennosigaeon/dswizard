from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, List, Tuple, Dict

import networkx as nx
from ConfigSpace import Configuration

from dswizard.core.model import CandidateStructure, CandidateId, Result
from dswizard.core.model import PartialConfig
from dswizard.util.util import prefixed_name

if TYPE_CHECKING:
    from dswizard.pipeline.pipeline import FlexiblePipeline


class JsonResultLogger:
    def __init__(self, directory: str, init: bool = True, overwrite: bool = False):
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

        self.directory = directory
        self.structure_fn = os.path.join(directory, 'structures.json')
        self.results_fn = os.path.join(directory, 'results.json')
        self.structure_ids = set()

        if init:
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

    def new_structure(self, structure: CandidateStructure, draw_structure: bool = False) -> None:
        if structure.cid.without_config() not in self.structure_ids:
            self.structure_ids.add(structure.cid.without_config())
            with open(self.structure_fn, 'a') as fh:
                fh.write(json.dumps(structure.as_dict()))
                fh.write('\n')

            # Results may already be created during structure creation
            for idx, result in enumerate(structure.results):
                self.log_evaluated_config(structure.cid.with_config(idx), result)

            if draw_structure:
                G = structure.pipeline.to_networkx()
                H = nx.nx_agraph.to_agraph(G)
                H.draw('{}/{}.png'.format(self.directory, structure.cid), prog='dot')

    def log_evaluated_config(self, cid: CandidateId, result: Result) -> None:
        if cid.without_config() not in self.structure_ids:
            # should never happen!
            raise ValueError('Unknown structure {}'.format(cid.without_config()))
        with open(self.results_fn, 'a') as fh:
            fh.write(
                json.dumps([cid.as_tuple(), result.as_dict() if result is not None else None])
            )
            fh.write("\n")

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

        self.file = '{}.json'.format(self.prefix)
        with open(self.file, 'w'):
            pass

    def new_step(self, name: str, config: PartialConfig) -> None:
        self.partial_configs.append(config)
        with open(self.file, 'a') as fh:
            fh.write(json.dumps([name, config.as_dict()]))
            fh.write('\n')

    def get_config(self, pipeline: FlexiblePipeline) -> Configuration:
        return self._merge_configs(self.partial_configs, pipeline)

    def restore_config(self, pipeline: FlexiblePipeline) -> Tuple[Configuration, List[PartialConfig]]:
        partial_configs: List[PartialConfig] = []

        with open(self.file) as fh:
            for line in fh:
                name, partial_config = json.loads(line)
                partial_config = PartialConfig.from_dict(partial_config)

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
            # noinspection PyTypeChecker
            partial_configs.append(PartialConfig(None, config, name, latest_mf))
        return partial_configs

    def _merge_configs(self, partial_configs: List[PartialConfig], pipeline: FlexiblePipeline) -> Configuration:
        complete = {}
        for partial_config in partial_configs:
            for param, value in partial_config.config.get_dictionary().items():
                param = prefixed_name(partial_config.name, param)
                complete[param] = value

        try:
            config = Configuration(pipeline.configuration_space, complete)
            return config
        except ValueError as ex:
            self.logger.error('Failed to reconstruct global config. '
                              'Config: {}\nConfigSpace: {}'.format(complete, pipeline.configuration_space))
            raise ex
