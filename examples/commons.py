"""
Worker for Examples 1-4
=======================

This class implements a very simple worker used in the firt examples.
"""

import time

import ConfigSpace as CS
import numpy

from hpbandster.core.worker import Worker


class MyWorker(Worker):

    def __init__(self, *args, sleep_interval=0, **kwargs):
        super().__init__(*args, **kwargs)

        self.sleep_interval = sleep_interval

    def compute(self,
                config: dict,
                config_info: dict,
                budget: float,
                result: dict,
                **kwargs: dict) -> dict:
        """
        Simple example for a compute function
        The loss is just a the config + some noise (that decreases with the budget)

        For dramatization, the function can sleep for a given interval to emphasizes
        the speed ups achievable with parallel workers.

        Args:
            config: dictionary containing the sampled configurations by the optimizer
            budget: (float) amount of time/epochs/etc. the model can use to train

        Returns:
            dictionary with mandatory fields:
                'loss' (scalar)
                'info' (dict)
        """

        res = numpy.clip(config['x'] + numpy.random.randn() / budget, config['x'] / 2, 1.5 * config['x'])
        time.sleep(self.sleep_interval)

        result['loss'] = float(res),  # this is the a mandatory field to run hyperband
        result['info'] = res  # can be used for any user-defined information - also mandatory

    @staticmethod
    def get_configspace():
        config_space = CS.ConfigurationSpace()
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('x', lower=0, upper=1))
        return config_space
