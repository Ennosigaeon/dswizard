import logging
from typing import Optional, Tuple

from ConfigSpace.configuration_space import ConfigurationSpace

from hpbandster.core.model import Structure
from hpbandster.core.base_config_generator import BaseConfigGenerator


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

    def get_config_space(self) -> Tuple[ConfigurationSpace, Structure]:
        """
        Sample a ConfigurationSpace and according Structure tuple

        :return:
        """
        raise NotImplementedError('_advance_to_next_stage not implemented for {}'.format(type(self).__name__))
