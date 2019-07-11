from typing import Dict, Union

from ConfigSpace.configuration_space import Configuration
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from hpbandster.components.base import ComponentChoice, EstimatorComponent


class FlexiblePipeline(Pipeline, BaseEstimator):

    def __init__(self, steps: Dict[str, Union[ComponentChoice, EstimatorComponent]], dataset_properties: dict):
        super().__init__(list(steps.items()))
        self.configuration = None
        self.dataset_properties = dataset_properties

    def fit(self, X, y=None, **fit_params):
        # TODO implement pipeline fitting
        # Was ist mit splits und merge in pipelines?

        super().fit(X, y, **fit_params)

    def set_hyperparameters(self, configuration, init_params=None):
        self.configuration = configuration

        for node_idx, (node_name, node) in enumerate(self.steps):
            sub_configuration_space = node.get_hyperparameter_search_space(
                dataset_properties=self.dataset_properties
            )
            sub_config_dict = {}
            for param in configuration:
                if param.startswith('%s:' % node_name):
                    value = configuration[param]
                    new_name = param.replace('%s:' % node_name, '', 1)
                    sub_config_dict[new_name] = value

            sub_configuration = Configuration(sub_configuration_space, values=sub_config_dict)

            if init_params is not None:
                sub_init_params_dict = {}
                for param in init_params:
                    if param.startswith('%s:' % node_name):
                        value = init_params[param]
                        new_name = param.replace('%s:' % node_name, '', 1)
                        sub_init_params_dict[new_name] = value
            else:
                sub_init_params_dict = None

            if isinstance(node, (ComponentChoice, EstimatorComponent)):
                node.set_hyperparameters(configuration=sub_configuration.get_dictionary(),
                                         init_params=sub_init_params_dict)
            else:
                raise NotImplementedError('Not supported yet!')

        return self
