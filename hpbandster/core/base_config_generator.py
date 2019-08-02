import logging
from typing import Tuple

from ConfigSpace.configuration_space import Configuration, ConfigurationSpace

from hpbandster.core.model import Job, ConfigInfo, Structure


class BaseConfigGenerator(object):
    """
    The config generator determines how new configurations are sampled. This can take very different levels of
    complexity, from random sampling to the construction of complex empirical prediction models for promising
    configurations.
    """

    def __init__(self,
                 configspace: ConfigurationSpace,
                 structure: Structure = None,
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
        self.structure = structure

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
