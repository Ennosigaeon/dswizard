from __future__ import annotations

import abc
import inspect
import logging
from typing import TYPE_CHECKING

from automl.components.base import EstimatorComponent, TunablePredictor, TunableEstimator
from dswizard.util import util

if TYPE_CHECKING:
    from dswizard.core.model import CandidateStructure, Result


class BaseStructureGenerator(abc.ABC):
    """
    The structure generator determines a pipeline structure before sampling new configurations. This can take very
    different levels of complexity, from dummy pipelines, static pipelines, random sampling to the construction of
    complex empirical prediction models for promising structures.
    """

    def __init__(self, dataset_properties: dict = None, timeout: int = None, logger: logging.Logger = None):
        """
        :param logger: for some debug output
        """
        # TODO move timeout somewhere more reasonable
        self.dataset_properties = dataset_properties
        self.timeout = timeout
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

    def register_result(self, candidate: CandidateStructure, result: Result, update_model: bool = True,
                        **kwargs) -> None:
        """
        registers finished runs

        Every time a run has finished, this function should be called to register it with the result logger. If
        overwritten, make sure to call this method from the base class to ensure proper logging.

        :param candidate: contains all necessary information about the job
        :param result:
        :param update_model: determines whether a model inside the config_generator should be updated
        :return:
        """

        if result.status == 'CRASHED':
            self.logger.warning('candidate {} failed'.format(candidate.id))

    @staticmethod
    def _get_estimator_instance(clazz: str) -> EstimatorComponent:
        try:
            return util.get_object(clazz)
        except TypeError:
            estimator = util.get_type(clazz)
            if 'predict' in inspect.getmembers(estimator, inspect.isfunction):
                # noinspection PyTypeChecker
                return TunablePredictor(estimator)
            else:
                # noinspection PyTypeChecker
                return TunableEstimator(estimator)
