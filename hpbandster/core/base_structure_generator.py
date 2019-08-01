import logging
from typing import Tuple, Optional

from ConfigSpace.configuration_space import Configuration, ConfigurationSpace

from hpbandster.core.base_config_generator import BaseConfigGenerator
from hpbandster.core.model import Job, ConfigInfo


class BaseStructureGenerator(object):
    """
    The structure generator determines a pipeline structure before sampling new configurations. This can take very
    different levels of complexity, from dummy pipelines, static pipelines, random sampling to the construction of
    complex empirical prediction models for promising structures.
    """

    def __init__(self, logger: logging.Logger = None):
        """
        :param logger: for some debug output
        """
        self.configspace: Optional[ConfigurationSpace] = None
        self.config_generator: Optional[BaseConfigGenerator] = None
        if logger is None:
            self.logger = logging.getLogger('StructureGenerator')
        else:
            self.logger = logger

    def set_config_generator(self, config_generator: BaseConfigGenerator) -> None:
        """
        function to set the ConfigGenerator used in this StructureGenerator. The StructureGenerator is responsible for
        providing a suited ConfigurationSpace for the ConfigGenerator.

        :param config_generator:
        """
        self.config_generator = config_generator

    def get_config(self, budget: float) -> Tuple[Configuration, ConfigInfo]:
        """
        function to sample a new configuration

        This function is called inside Hyperband to query a new configuration
        :param budget: the budget for which this configuration is scheduled
        :return: must return a valid configuration and a (possibly empty) info dict
        """

        raise NotImplementedError('This function needs to be overwritten in {}.'.format(self.__class__.__name__))

    def new_result(self, job: Job, update_model: bool = True) -> None:
        """
        registers finished runs

        Every time a run has finished, this function should be called to register it with the result logger. If
        overwritten, make sure to call this method from the base class to ensure proper logging.

        :param job: contains all necessary information about the job
        :param update_model: determines whether a model inside the config_generator should be updated
        :return:
        """

        if job.exception is not None:
            self.logger.warning('job {} failed with exception\n{}'.format(job.id, job.exception))

    def get_config_space(self) -> ConfigurationSpace:
        """
        get the ConfigurationSpace for this specific pipeline

        :return:
        """
        raise self.configspace
