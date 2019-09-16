from __future__ import annotations

import abc
import logging
from typing import List, Callable, Optional, Tuple, TYPE_CHECKING

import Pyro4
from ConfigSpace import Configuration

from dswizard.core.config_cache import ConfigCache

if TYPE_CHECKING:
    from dswizard.core.base_config_generator import BaseConfigGenerator
    from dswizard.core.base_iteration import BaseIteration
    from dswizard.core.base_structure_generator import BaseStructureGenerator
    from dswizard.core.model import CandidateStructure, CandidateId
    from dswizard.components.pipeline import FlexiblePipeline


class BanditLearner(abc.ABC):

    def __init__(self,
                 run_id: str,
                 nameserver: str = None,
                 nameserver_port: int = None,
                 structure_generator: BaseStructureGenerator = None,
                 logger: logging.Logger = None):
        self.run_id = run_id
        self.nameserver = nameserver
        self.nameserver_port = nameserver_port

        self.structure_generator = structure_generator

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

    def optimize(self, starter: Callable[[CandidateId, CandidateStructure, Optional[Configuration]], None],
                 iteration_kwargs: dict) -> None:
        """
        Optimize all hyperparameters
        :param starter:
        :param iteration_kwargs:
        :return:
        """
        # noinspection PyTypeChecker
        for candidate, iteration in self._get_next_structure(iteration_kwargs):
            cg = self._get_config_generator(candidate.pipeline)
            cg.optimize(starter, candidate)

            self.iterations[iteration].register_result(candidate)
            self.structure_generator.new_result(candidate)

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

    def _get_config_generator(self, pipeline: FlexiblePipeline) -> Optional[BaseConfigGenerator]:
        cache: Optional[ConfigCache] = None
        if self.nameserver is None:
            cache = ConfigCache.instance()
        else:
            with Pyro4.locateNS(host=self.nameserver, port=self.nameserver_port) as ns:
                uri = list(ns.list(prefix='{}.config_generator'.format(self.run_id)).values())
                if len(uri) != 1:
                    raise ValueError('Expected exactly one ConfigCache but found {}'.format(len(uri)))
                # noinspection PyTypeChecker
                cache = Pyro4.Proxy(uri[0])

        return cache.get_config_generator(pipeline.configuration_space)
