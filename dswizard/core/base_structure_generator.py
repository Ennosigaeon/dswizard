import abc
import logging

from dswizard.core.model import CandidateStructure


class BaseStructureGenerator(abc.ABC):
    """
    The structure generator determines a pipeline structure before sampling new configurations. This can take very
    different levels of complexity, from dummy pipelines, static pipelines, random sampling to the construction of
    complex empirical prediction models for promising structures.
    """

    def __init__(self, logger: logging.Logger = None):
        """
        :param logger: for some debug output
        """
        if logger is None:
            self.logger = logging.getLogger('StructureGenerator')
        else:
            self.logger = logger

    @abc.abstractmethod
    def get_candidate(self, budget: float) -> CandidateStructure:
        """
        Sample a ConfigurationSpace and according Structure tuple

        :return:
        """
        raise NotImplementedError('get_config_space not implemented for {}'.format(type(self).__name__))

    def new_result(self, candidate: CandidateStructure, update_model: bool = True) -> None:
        """
        registers finished runs

        Every time a run has finished, this function should be called to register it with the result logger. If
        overwritten, make sure to call this method from the base class to ensure proper logging.

        :param candidate: contains all necessary information about the job
        :param update_model: determines whether a model inside the config_generator should be updated
        :return:
        """

        if candidate.status == 'CRASHED':
            self.logger.warning('candidate {} failed'.format(candidate.id))
