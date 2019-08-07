import logging
from typing import List, Optional, Tuple, Callable

from dswizard.core.base_config_generator import BaseConfigGenerator
from dswizard.core.base_iteration import BaseIteration
from dswizard.core.model import ConfigId, Datum, Job


class HPO:

    def __init__(self, config_generator: BaseConfigGenerator = None, logger: logging.Logger = None):
        self.config_generator = config_generator

        if logger is None:
            self.logger = logging.getLogger('Racing')
        else:
            self.logger = logger

        self.iterations: List[BaseIteration] = []
        self.config = {}
        self.max_iterations = 0

    def get_next_iteration(self, iteration: int, iteration_kwargs: dict) -> BaseIteration:
        """
        instantiates the next iteration

        Overwrite this to change the iterations for different optimizers
        :param iteration: the index of the iteration to be instantiated
        :param iteration_kwargs: additional kwargs for the iteration class. Defaults to empty dictionary
        :return: a valid HB iteration object
        """

        raise NotImplementedError('implement get_next_iteration for {}'.format(type(self).__name__))

    def active_iterations(self) -> List[int]:
        """
        function to find active (not marked as finished) iterations
        :return: all active iteration indices (empty if there are none)
        """

        return list(filter(lambda idx: not self.iterations[idx].is_finished, range(len(self.iterations))))

    def register_result(self, job: Job) -> None:
        self.iterations[job.id.iteration].register_result(job)
        self.config_generator.new_result(job)

    def optimize(self, starter: Callable[[ConfigId, Datum], None], iteration_kwargs: dict):
        """
        Optimize all hyperparameters
        :param starter:
        :param iteration_kwargs:
        :return:
        """
        for id, datum in self._get_next_datum(iteration_kwargs):
            starter(id, datum)

    def _get_next_datum(self, iteration_kwargs: dict = None) -> Optional[Tuple[ConfigId, Datum]]:
        n_iterations = self.max_iterations
        while True:
            next_run = None
            # find a new run to schedule
            for i in self.active_iterations():
                next_run = self.iterations[i].get_next_run()
                if next_run is not None:
                    break

            if next_run is not None:
                yield next_run
            else:
                if n_iterations > 0:  # we might be able to start the next iteration
                    iteration = len(self.iterations)
                    self.logger.info('Starting iteration {}'.format(iteration))
                    self.iterations.append(self.get_next_iteration(iteration, iteration_kwargs))
                    n_iterations -= 1
                else:
                    # Done
                    break
