from typing import Dict, Type

from ConfigSpace import ConfigurationSpace

from dswizard.core.base_config_generator import BaseConfigGenerator
from dswizard.core.model import Structure, Job


class ConfigGeneratorCache:

    def __init__(self, clazz: Type[BaseConfigGenerator], init_args: dict):
        self.clazz = clazz
        self.init_args = init_args

        self.cache: Dict[ConfigurationSpace, BaseConfigGenerator] = {}

    def get(self, configspace: ConfigurationSpace, structure: Structure) -> BaseConfigGenerator:
        if configspace not in self.cache:
            cg = self.clazz(configspace, structure, **self.init_args)
            self.cache[configspace] = cg
        return self.cache[configspace]

    def register_result(self, job: Job):
        try:
            # TODO only workaround, job should contain only structure and not configspace
            self.cache[job.configspace].register_result(job)
        except KeyError:
            # Should never happen
            pass
