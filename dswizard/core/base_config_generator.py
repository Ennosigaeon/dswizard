from __future__ import annotations

import abc
from typing import Dict, Any

from ConfigSpace.configuration_space import ConfigurationSpace, Configuration

from dswizard.core.model import StatusType


class BaseConfigGenerator(abc.ABC):
    """
    The config generator determines how new configurations are sampled. This can take very different levels of
    complexity, from random sampling to the construction of complex empirical prediction models for promising
    configurations.
    """

    def __init__(self,
                 configspace: ConfigurationSpace,
                 **kwargs):
        """
        :param configspace:
        :param working_directory:
        :param logger: for some debug output
        """
        if configspace is None:
            raise ValueError('You have to provide a valid ConfigSpace object')

        self.configspace: ConfigurationSpace = configspace
        self.expected_size = self.configspace.get_default_configuration().get_array().size

        self.explanations = {}

    @abc.abstractmethod
    def sample_config(self, default: bool = False, **kwargs) -> Configuration:
        pass

    def register_result(self, config: Configuration, loss: float, status: StatusType, update_model: bool = True,
                        **kwargs) -> None:
        """
        registers finished runs

        Every time a run has finished, this function should be called to register it with the result logger. If
        overwritten, make sure to call this method from the base class to ensure proper logging.

        :param config:
        :param loss:
        :param status:
        :param update_model: determines whether a model inside the config_generator should be updated
        :return:
        """
        pass

    def explain(self) -> Dict[str, Any]:
        return self.explanations
