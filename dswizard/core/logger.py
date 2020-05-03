from __future__ import annotations

import json
import os
import pickle
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
        if structure.id not in self.structure_ids:
            self.structure_ids.add(structure.id)
            with open(self.structure_fn, 'a') as fh:
                fh.write(json.dumps(structure.as_dict()))
                fh.write('\n')

            if draw_structure:
                G = structure.pipeline.to_networkx()
                H = nx.nx_agraph.to_agraph(G)
                H.draw('{}/{}.png'.format(self.directory, structure.id), prog='dot')

    def log_evaluated_config(self, job: Job) -> None:
        if job.id.without_config() not in self.structure_ids:
            # should never happen! TODO: log warning here!
            self.structure_ids.add(job.id)
            with open(self.structure_fn, 'a') as fh:
                fh.write(json.dumps([job.id.as_tuple(), job.config.get_dictionary(), {}]))
                fh.write('\n')
        with open(self.results_fn, 'a') as fh:
            fh.write(
                json.dumps([job.id.as_tuple(), job.budget, job.result.as_dict() if job.result is not None else None])
            )
            fh.write("\n")


class ProcessLogger:

    def __init__(self, directory: str, config_id):

        os.makedirs(directory, exist_ok=True)
        self.prefix = os.path.join(directory, '{}-{}-{}'.format(*config_id.as_tuple()))

        self.file = '{}.json'.format(self.prefix)
        with open(self.file, 'w'):
            pass

    def new_step(self, name: str, config: PartialConfig) -> None:
        with open(self.file, 'a') as fh:
            fh.write(json.dumps([name, config.as_dict()]))
            fh.write('\n')
        with open('{}-{}.pickle'.format(self.prefix, name), 'wb') as fh:
            pickle.dump(config.meta, fh)

    def restore_config(self, pipeline: FlexiblePipeline) -> Tuple[Configuration, List[PartialConfig]]:
        complete = {}
        partial_configs: List[PartialConfig] = []

        missing_steps = set(pipeline.all_names())
        with open(self.file) as fh:
            for line in fh:
                name, partial_config = json.loads(line)
                partial_config = PartialConfig.from_dict(partial_config)

                pickle_file = '{}-{}.pickle'.format(self.prefix, name)
                with open(pickle_file, 'rb') as fh2:
                    partial_config.meta = pickle.load(fh2)
                os.remove(pickle_file)

                for param, value in partial_config.configuration.get_dictionary().items():
                    param = prefixed_name(name, param)
                    complete[param] = value

                missing_steps.remove(name)
                partial_configs.append(partial_config)

        # Create random configuration for missing steps
        for name in missing_steps:
            config = pipeline.get_step(name).get_hyperparameter_search_space(pipeline.dataset_properties) \
                .sample_configuration()
            for param, value in config.get_dictionary().items():
                param = prefixed_name(name, param)
                complete[param] = value

        config = Configuration(pipeline.configuration_space, complete)
        os.remove(self.file)
        return config, partial_configs
