import os

import json
from ConfigSpace import Configuration


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

    def new_structure(self, structure) -> None:
        if structure.id not in self.structure_ids:
            self.structure_ids.add(structure.id)
            with open(self.structure_fn, 'a') as fh:
                fh.write(json.dumps(structure.as_dict()))
                fh.write('\n')

    def log_evaluated_config(self, job) -> None:
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
