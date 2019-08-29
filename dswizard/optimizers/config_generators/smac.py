from typing import List

from ConfigSpace import Configuration, ConfigurationSpace
from smac.facade import smac_facade
from smac.scenario.scenario import Scenario

from dswizard.components.pipeline import FlexiblePipeline
from dswizard.core.base_config_generator import BaseConfigGenerator
from dswizard.core.model import Job


class SMAC(BaseConfigGenerator):

    def __init__(self,
                 configspace: ConfigurationSpace,
                 pipeline: FlexiblePipeline = None,
                 num_samples: int = 20,
                 **kwargs):
        super().__init__(configspace, pipeline, **kwargs)

        scenario = Scenario({"run_obj": "quality",
                             "cs": configspace,
                             "deterministic": "true",
                             "initial_incumbent": "DEFAULT",
                             "output_dir": "/tmp/smac",
                             "acq_opt_challengers": num_samples
                             })
        self.smac: smac_facade.SMAC = smac_facade.SMAC(scenario, lambda x: None)
        self.challengers: List[Configuration] = []
        self.idx = 0

    def get_config(self, budget: float = None) -> Configuration:
        if self.idx >= len(self.challengers):
            self.idx = 0

            X, y = self.smac.solver.rh2EPM.transform(self.smac.solver.runhistory)
            self.challengers = list(self.smac.solver.choose_next(X, y))

        self.idx += 1
        return self.challengers[self.idx - 1]

    def get_config_for_step(self, step: str, budget: float = None) -> Configuration:
        raise NotImplementedError('SMAC does not support JIT configuration')

    def register_result(self, job: Job, update_model: bool = True) -> None:
        super().register_result(job, update_model)
        self.smac.solver.runhistory.add(job.config, job.result.loss, job.result.runtime, job.result.status, str(job.id))
