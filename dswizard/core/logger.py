from __future__ import annotations

import json
import logging
import os
from typing import TYPE_CHECKING, List, Tuple

import networkx as nx
from ConfigSpace import Configuration

from dswizard.core.model import PartialConfig
from dswizard.util.util import prefixed_name

if TYPE_CHECKING:
    from dswizard.core.model import CandidateStructure, Job
    from dswizard.components.pipeline import FlexiblePipeline


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

        self.directory = directory
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

    def new_structure(self, structure: CandidateStructure, draw_structure: bool = False) -> None:
        if structure.cid not in self.structure_ids:
            self.structure_ids.add(structure.cid)
            with open(self.structure_fn, 'a') as fh:
                fh.write(json.dumps(structure.as_dict()))
                fh.write('\n')

            if draw_structure:
                G = structure.pipeline.to_networkx()
                H = nx.nx_agraph.to_agraph(G)
                H.draw('{}/{}.png'.format(self.directory, structure.cid), prog='dot')

    def log_evaluated_config(self, job: Job) -> None:
        if job.cid.without_config() not in self.structure_ids:
            # should never happen!
            self.structure_ids.add(job.cid)
            with open(self.structure_fn, 'a') as fh:
                fh.write(json.dumps([job.cid.as_tuple(), job.config.get_dictionary(), {}]))
                fh.write('\n')
        with open(self.results_fn, 'a') as fh:
            fh.write(
                json.dumps([job.cid.as_tuple(), job.result.as_dict() if job.result is not None else None])
            )
            fh.write("\n")


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

        config = self._merge_configs(partial_configs, pipeline)
        os.remove(self.file)
        return config, partial_configs

    def _merge_configs(self, partial_configs: List[PartialConfig], pipeline: FlexiblePipeline) -> Configuration:
        if len(partial_configs) == 0:
            self.logger.warning('Encountered job without any partial configurations. Simulating complete config')

        complete = {}
        missing_steps = set(pipeline.all_names())

        for partial_config in partial_configs:
            for param, value in partial_config.config.get_dictionary().items():
                param = prefixed_name(partial_config.name, param)
                complete[param] = value
            missing_steps.remove(partial_config.name)

        # Create random configuration for missing steps
        latest_mf = None if len(partial_configs) == 0 else partial_configs[-1].mf
        for name in missing_steps:
            config = pipeline.get_step(name).get_hyperparameter_search_space(mf=latest_mf) \
                .sample_configuration()
            for param, value in config.get_dictionary().items():
                param = prefixed_name(name, param)
                complete[param] = value

        try:
            return Configuration(pipeline.configuration_space, complete)
        except ValueError as ex:
            self.logger.error('Failed to reconstruct global config. '
                              'Config: {}\nConfigSpace: {}'.format(complete, pipeline.configuration_space))
            raise ex
