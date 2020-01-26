from __future__ import annotations

import abc
import logging
from typing import List, Callable, Optional, Tuple, TYPE_CHECKING

from ConfigSpace import Configuration
from ConfigSpace.configuration_space import ConfigurationSpace

from dswizard import utils

if TYPE_CHECKING:
    from dswizard.core.base_config_generator import BaseConfigGenerator
    from dswizard.core.base_iteration import BaseIteration
    from dswizard.core.base_structure_generator import BaseStructureGenerator
    from dswizard.core.model import CandidateStructure, CandidateId, Dataset, MetaFeatures, Job


class BanditLearner(abc.ABC):

    def __init__(self,
                 run_id: str,
                 nameserver: str = None,
                 nameserver_port: int = None,
                 structure_generator: BaseStructureGenerator = None,
                 sample_config: bool = True,
                 logger: logging.Logger = None):
        self.run_id = run_id
        self.nameserver = nameserver
        self.nameserver_port = nameserver_port

        self.structure_generator = structure_generator
        self.sample_config = sample_config

        if logger is None:
            self.logger = logging.getLogger('Racing')
        else:
            self.logger = logger

        self.iterations: List[BaseIteration] = []
        self.config = {}
        self.max_iterations = 0

    @abc.abstractmethod
    def _get_next_iteration(self, iteration: int, iteration_kwargs: dict) -> BaseIteration:
        """
        instantiates the next iteration

        Overwrite this to change the iterations for different optimizers
        :param iteration: the index of the iteration to be instantiated
        :param iteration_kwargs: additional kwargs for the iteration class. Defaults to empty dictionary
        :return: a valid HB iteration object
        """
        pass

    def optimize(self, starter: Callable[[Dataset, CandidateId, CandidateStructure, Optional[Configuration]], None],
                 ds: Dataset, iteration_kwargs: dict, iterations: int = 1) -> None:
        """
        Optimize all hyperparameters
        :param starter:
        :param ds:
        :param iterations:
        :param iteration_kwargs:
        :return:
        """
        # noinspection PyTypeChecker
        for candidate, iteration in self._get_next_structure(iteration_kwargs):
            # Optimize hyperparameters
            for i in range(iterations):
                config_id = candidate.id.with_config(i)
                if self.sample_config:
                    cg = self._get_config_generator(candidate.budget, candidate.pipeline.configuration_space,
                                                    ds.meta_features)
                    config = cg.sample_config()
                    starter(ds, config_id, candidate, config)
                else:
                    starter(ds, config_id, candidate, None)

    def _get_next_structure(self, iteration_kwargs: dict = None) -> List[Tuple[CandidateStructure, int]]:
        n_iterations = self.max_iterations
        while True:
            next_candidate = None
            # find a new run to schedule
            for i in filter(lambda idx: not self.iterations[idx].is_finished, range(len(self.iterations))):
                next_candidate = self.iterations[i].get_next_candidate()
                if next_candidate is not None:
                    break

            if next_candidate is not None:
                # noinspection PyUnboundLocalVariable
                yield next_candidate, i
            else:
                if n_iterations > 0:  # we might be able to start the next iteration
                    iteration = len(self.iterations)
                    self.logger.info('Starting iteration {}'.format(iteration))
                    self.iterations.append(self._get_next_iteration(iteration, iteration_kwargs))
                    n_iterations -= 1
                else:
                    # Done
                    break

    def _get_config_generator(self, budget: float, configspace: ConfigurationSpace, meta_features: MetaFeatures) -> \
            Optional[BaseConfigGenerator]:
        cache = utils.get_config_generator_cache(self.nameserver, self.nameserver_port, self.run_id)
        return cache.get_config_generator(budget, configspace, meta_features)

    def register_result(self, job: Job, update_model: bool = True):
        self.iterations[-1].register_result(job.cs)
        self.structure_generator.register_result(job.cs, job.result, update_model=update_model)
