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
        self.run_infos: typing.Dict[str, typing.Tuple[RunInfo, float]] = {}

    def sample_config(self, recursion_depth: int = 5) -> Configuration:
        # TODO even though initial_incumbent is set to DEFAULT, the first configuration is random

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
            if recursion_depth > 0:
                return self.sample_config(recursion_depth - 1)
            else:
                self.logger.warning('Repeatedly failed to sample configuration. Using random configuration instead.')
                config = self.config_space.sample_configuration()
                config.origin = 'Random Search'
                run_info = RunInfo(config, run_info.instance,
                                   run_info.instance_specific, run_info.seed, run_info.cutoff, run_info.capped,
                                   run_info.budget, run_info.source_id)

        if run_info.config is None:
            config = self.config_space.sample_configuration()
            config.origin = 'Random Search'
        else:
            config = run_info.config

        self.runhistory.add(
            config=config,
            cost=float(MAXINT),
            time=0.0,
            status=SmacStatus.RUNNING,
            instance_id=run_info.instance,
            seed=run_info.seed,
            budget=run_info.budget,
        )

        self.stats.submitted_ta_runs += 1

        self.run_infos[str(config)] = (run_info, time.time())
        return config

    def register_result(self, config: Configuration, loss: float, status: StatusType, update_model: bool = True,
                        **kwargs) -> None:
        try:
            key = str(config)
            info, start_time = self.run_infos[key]
            end_time = time.time()
            del self.run_infos[key]
            result = RunValue(loss, end_time - start_time, SmacStatus(status.value), start_time, end_time, {})
            if update_model:
                self._incorporate_run_results(info, result, 10)
        except KeyError:
            # Configuration was sampled during structure search and not via SMAC
            if update_model:
                self.runhistory.add(
                    config=config,
                    cost=loss,
                    time=0.0,
                    status=SmacStatus(status.value),
                    instance_id=None,
                    seed=0,
                )


class SmacGenerator(BaseConfigGenerator):

    def __init__(self, configspace: ConfigurationSpace, working_directory: str = '.'):
        super().__init__(configspace)

        self.working_directory = os.path.join(working_directory, f'smac/{random.randint(0, 10000000):d}/')

        smac_logger = logging.getLogger('smac')
        logging.getLogger('smac').setLevel(logging.WARNING)

        scenario = Scenario({
            'run_obj': 'quality',
            'deterministic': True,
            'shared-model': False,
            'cs': self.configspace,
            'output_dir': self.working_directory
        })
        scenario.logger = smac_logger

        self.smbo: SplitSMBO = SMAC4HPO(scenario=scenario, smbo_class=SplitSMBO).solver
        self.smbo.logger = smac_logger
        self.smbo.start()

    def sample_config(self, default: bool = False, **kwargs) -> Configuration:
        if default:
            config = self.configspace.get_default_configuration()
            config.origin = 'Default'
            return config

        config = self.smbo.sample_config()
        return config

    def register_result(self, config: Configuration, loss: float, status: StatusType, update_model: bool = True,
                        **kwargs) -> None:
        self.smbo.register_result(config, loss, status)
