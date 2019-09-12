from __future__ import annotations

import abc
import logging
from typing import Callable, Optional, TYPE_CHECKING

from ConfigSpace.configuration_space import ConfigurationSpace, Configuration

from dswizard.core.model import StatusType

if TYPE_CHECKING:
    from dswizard.core.model import Job, CandidateStructure, CandidateId


class BaseConfigGenerator(abc.ABC):
    """
    The config generator determines how new configurations are sampled. This can take very different levels of
    complexity, from random sampling to the construction of complex empirical prediction models for promising
    configurations.
    """

    def __init__(self,
                 configspace: ConfigurationSpace,
                 on_the_fly_generation: bool = False,
                 logger: logging.Logger = None):
        """
        :param logger: for some debug output
        """
        if configspace is None:
            raise ValueError('You have to provide a valid ConfigSpace object')

        if logger is None:
            self.logger = logging.getLogger('ConfigGenerator')
        else:
            self.logger = logger
        self.configspace: ConfigurationSpace = configspace
        self.on_the_fly_generation = on_the_fly_generation
        self.cs: Optional[CandidateStructure] = None

    def optimize(self,
                 starter: Callable[[CandidateId, CandidateStructure, Optional[Configuration]], None],
                 candidate: CandidateStructure,
                 iterations: int = 1):
        self.cs = candidate
        for i in range(iterations):
            config_id = candidate.id.with_config(i)
            if self.on_the_fly_generation:
                starter(config_id, candidate, None)
            else:
                config = self.get_config()
                starter(config_id, candidate, config)

    @abc.abstractmethod
    def get_config(self, budget: float = None) -> Configuration:
        pass

    @abc.abstractmethod
    def get_config_for_step(self, step: str, budget: float = None) -> Configuration:
        pass

    def register_result(self, job: Job, update_model: bool = True) -> None:
        """
        registers finished runs

        Every time a run has finished, this function should be called to register it with the result logger. If
        overwritten, make sure to call this method from the base class to ensure proper logging.

        :param job: contains all necessary information about the job
        :param update_model: determines whether a model inside the config_generator should be updated
        :return:
        """

        if job.result.status is not StatusType.SUCCESS:
            self.logger.warning('job {} failed with {}'.format(job.id, job.result.status))
        self.cs.add_result(job.budget, job.result)
