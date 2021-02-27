import logging
import os
import random
import threading
from queue import Queue

try:
    from ConfigSpace.configuration_space import Configuration, ConfigurationSpace
    from smac.facade.smac_facade import SMAC
    from smac.scenario.scenario import Scenario
    from smac.stats.stats import Stats
    from smac.tae.execute_ta_run import ExecuteTARun, StatusType as SmacStatus
except ImportError:
    import sys
    print("smac is not installed. See https://pypi.org/project/smac/0.8.0/ for installing smac", file=sys.stderr)
    sys.exit(1)

from dswizard.core.base_config_generator import BaseConfigGenerator
from dswizard.core.model import StatusType


class GeneratingTARun(ExecuteTARun):

    def __init__(self):
        super().__init__([], run_obj='quality')
        self.configs = Queue()
        self.results = Queue()

    def run(self, config: Configuration, instance: str,
            cutoff: int = None,
            seed: int = 12345,
            instance_specific: str = "0"):
        self.configs.put(config)

        # Wait for challenger to be processed
        status, cost, runtime, additional_info = self.results.get(block=True)

        self.runhistory.add(config=config,
                            cost=cost, time=runtime, status=status,
                            instance_id=instance, seed=seed,
                            additional_info=additional_info)
        return status, cost, runtime, additional_info


class CustomStats(Stats):

    def __init__(self, scenario: Scenario):
        super().__init__(scenario)
        self.shutdown = False

    def is_budget_exhausted(self):
        return self.shutdown


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
            'shared-model': True,
            'cutoff_time': 60,

            'cs': self.configspace,
            'initial_incumbent': 'DEFAULT',

            'input_psmac_dirs': self.working_directory + 'in/',
            'output_dir': self.working_directory + 'out/'
        })
        scenario.logger = smac_logger

        self.stats = CustomStats(scenario)
        self.stats._logger = smac_logger
        self.tae_run = GeneratingTARun()
        self.tae_run.logger = smac_logger

        self.smbo = SMAC(scenario=scenario, tae_runner=self.tae_run, stats=self.stats)
        self.smbo.logger = smac_logger

        self.thread = threading.Thread(target=self.smbo.optimize)
        self.thread.start()

    def sample_config(self, default: bool = False) -> Configuration:
        if default:
            return self.configspace.get_default_configuration()

        return self.tae_run.configs.get(block=True)

    def register_result(self, config: Configuration, loss: float, status: StatusType, update_model: bool = True,
                        **kwargs) -> None:
        self.tae_run.results.put((SmacStatus(status.value), loss, 0, {}))

    def __del__(self):
        self.stats.shutdown = True

        # Draw samples until next check for budget exhaustion is invoked
        while self.thread.is_alive():
            # noinspection PyTypeChecker
            self.register_result(None, 1, StatusType.CRASHED)
