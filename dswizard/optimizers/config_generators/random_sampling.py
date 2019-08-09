from typing import Callable

from ConfigSpace.configuration_space import Configuration

from dswizard.core.base_config_generator import BaseConfigGenerator
from dswizard.core.model import Structure, CandidateId, CandidateStructure


class RandomSampling(BaseConfigGenerator):
    """
    class to implement random sampling from a ConfigSpace
    """

    def optimize(self,
                 starter: Callable[[CandidateId, Configuration, CandidateStructure], None],
                 candidate: CandidateStructure,
                 iterations: int = 5):
        self.cs = candidate
        for i in range(iterations):
            config = self._get_config()
            config_id = candidate.id.with_config(i)
            starter(config_id, config, candidate)

    def _get_config(self) -> Configuration:
        if self.configspace is None:
            raise ValueError('No configuration space provided. Call set_config_space(ConfigurationSpace) first.')

        return self.configspace.sample_configuration()
