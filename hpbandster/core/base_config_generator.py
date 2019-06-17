import logging
from typing import Tuple

from hpbandster.core.model import Job


class BaseConfigGenerator(object):
    """
    The config generator determines how new configurations are sampled. This can take very different levels of
    complexity, from random sampling to the construction of complex empirical prediction models for promising
    configurations.
    """

    def __init__(self, logger: logging.Logger = None):
        """
        Parameters
        ----------
        logger: logging.logger
            for some debug output

        """

        if logger is None:
            self.logger = logging.getLogger('hpbandster')
        else:
            self.logger = logger

    def get_config(self, budget: float) -> Tuple[dict, dict]:
        """
        function to sample a new configuration

        This function is called inside Hyperband to query a new configuration

        Parameters
        ----------
        budget: float
            the budget for which this configuration is scheduled

        returns: (config, info_dict)
            must return a valid configuration and a (possibly empty) info dict
        """
        raise NotImplementedError('This function needs to be overwritten in {}.'.format(self.__class__.__name__))

    def new_result(self, job: Job, update_model: bool = True) -> None:
        """
        registers finished runs

        Every time a run has finished, this function should be called
        to register it with the result logger. If overwritten, make
        sure to call this method from the base class to ensure proper
        logging.


        Parameters
        ----------
        job: instance of hpbandster.distributed.dispatcher.Job
            contains all necessary information about the job
        update_model: boolean
            determines whether a model inside the config_generator should be updated
        """
        if job.exception is not None:
            self.logger.warning("job {} failed with exception\n{}".format(job.id, job.exception))
