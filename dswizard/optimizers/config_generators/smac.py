from typing import Tuple, List

from ConfigSpace import Configuration, ConfigurationSpace
from smac.facade import smac_facade
from smac.scenario.scenario import Scenario

from dswizard.core.base_config_generator import BaseConfigGenerator
from dswizard.core.model import ConfigInfo, Structure, Job


class SMAC(BaseConfigGenerator):

    def __init__(self,
                 configspace: ConfigurationSpace,
                 structure: Structure = None,
                 num_samples: int = 20,
                 **kwargs):
        super().__init__(configspace, structure, **kwargs)

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

    def get_config(self, budget: float) -> Tuple[Configuration, ConfigInfo]:
        if self.idx >= len(self.challengers):
            self.idx = 0

            X, y = self.smac.solver.rh2EPM.transform(self.smac.solver.runhistory)
            self.challengers = list(self.smac.solver.choose_next(X, y))

        info = ConfigInfo(
            model_based_pick=True,
            structure=self.structure
        )
        self.idx += 1
        return self.challengers[self.idx - 1], info

    def new_result(self, job: Job, update_model: bool = True) -> None:
        super().new_result(job, update_model)
        self.smac.solver.runhistory.add(job.config, job.result.loss, job.result.time, job.result.status, str(job.id))
