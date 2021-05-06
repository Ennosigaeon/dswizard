import logging
import os
import random
import time
import typing

from ConfigSpace.configuration_space import Configuration, ConfigurationSpace
from smac.facade.smac_hpo_facade import SMAC4HPO
from smac.intensification.abstract_racer import RunInfoIntent
from smac.optimizer.smbo import SMBO
from smac.runhistory.runhistory import RunValue, RunInfo
from smac.scenario.scenario import Scenario
from smac.tae.serial_runner import StatusType as SmacStatus
from smac.utils.constants import MAXINT

from dswizard.core.base_config_generator import BaseConfigGenerator
from dswizard.core.model import StatusType


class SplitSMBO(SMBO):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.run_infos: typing.Dict[Configuration, typing.Tuple[RunInfo, float]] = {}

    def sample_config(self) -> Configuration:
        # sample next configuration for intensification
        # Initial design runs are also included in the BO loop now.
        intent, run_info = self.intensifier.get_next_run(
            challengers=self.initial_design_configs,
            incumbent=self.incumbent,
            chooser=self.epm_chooser,
            run_history=self.runhistory,
            repeat_configs=self.intensifier.repeat_configs,
            num_workers=self.tae_runner.num_workers(),
        )

        # remove config from initial design challengers to not repeat it again
        self.initial_design_configs = [c for c in self.initial_design_configs if c != run_info.config]

        if intent == RunInfoIntent.SKIP:
            return self.sample_config()

        self.runhistory.add(
            config=run_info.config,
            cost=float(MAXINT),
            time=0.0,
            status=SmacStatus.RUNNING,
            instance_id=run_info.instance,
            seed=run_info.seed,
            budget=run_info.budget,
        )

        run_info.config.config_id = self.runhistory.config_ids[run_info.config]
        self.stats.submitted_ta_runs += 1

        self.run_infos[run_info.config] = (run_info, time.time())
        return run_info.config

    def register_result(self, config: Configuration, loss: float, status: StatusType, update_model: bool = True,
                        **kwargs) -> None:
        info, start_time = self.run_infos[config]
        end_time = time.time()
        del self.run_infos[config]
        result = RunValue(loss, end_time - start_time, SmacStatus(status.value), start_time, end_time, {})
        self._incorporate_run_results(info, result, 0)


class SmacGenerator(BaseConfigGenerator):

    def __init__(self, configspace: ConfigurationSpace, working_directory: str = '.'):
        super().__init__(configspace)

        self.working_directory = os.path.join(working_directory, 'smac/{:d}/'.format(random.randint(0, 10000000)))

        smac_logger = logging.getLogger('smac')
        logging.getLogger('smac').setLevel(logging.WARNING)

        scenario = Scenario({
            'abort_on_first_run_crash': True,
            'run_obj': 'quality',
            'deterministic': True,
            'shared-model': False,

            'cs': self.configspace,
            'initial_incumbent': 'DEFAULT',

            'output_dir': self.working_directory
        })
        scenario.logger = smac_logger

        self.smbo: SplitSMBO = SMAC4HPO(scenario=scenario, smbo_class=SplitSMBO).solver
        self.smbo.logger = smac_logger

    def sample_config(self, default: bool = False) -> Configuration:
        if default:
            return self.configspace.get_default_configuration()

        return self.smbo.sample_config()

    def register_result(self, config: Configuration, loss: float, status: StatusType, update_model: bool = True,
                        **kwargs) -> None:
        self.smbo.register_result(config, loss, status)
