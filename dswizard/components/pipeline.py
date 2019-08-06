from typing import Dict, List

import numpy as np
from ConfigSpace.configuration_space import Configuration, ConfigurationSpace
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from dswizard.components.base import ComponentChoice, EstimatorComponent
from dswizard.core.model import Structure


class FlexiblePipeline(Pipeline, BaseEstimator):

    def __init__(self, steps: Structure, dataset_properties: dict):
        super().__init__(list(steps.items()))
        self.steps_ = steps
        self.configuration = None
        self.dataset_properties = dataset_properties

    def fit(self, X, y=None, **fit_params):
        # TODO implement pipeline fitting
        # Was ist mit splits und merge in pipelines?

        super().fit(X, y, **fit_params)

    def set_hyperparameters(self, configuration: dict, init_params=None):
        self.configuration = configuration

        for node_idx, (node_name, node) in enumerate(self.steps):
            sub_configuration_space = node.get_hyperparameter_search_space(
                dataset_properties=self.dataset_properties
            )
            sub_config_dict = {}
            for param in configuration:
                if param.startswith('{}:'.format(node_name)):
                    value = configuration[param]
                    new_name = param.replace('{}:'.format(node_name), '', 1)
                    sub_config_dict[new_name] = value

            sub_configuration = Configuration(sub_configuration_space, values=sub_config_dict)

            if init_params is not None:
                sub_init_params_dict = {}
                for param in init_params:
                    if param.startswith('{}:'.format(node_name)):
                        value = init_params[param]
                        new_name = param.replace('{}:'.format(node_name), '', 1)
                        sub_init_params_dict[new_name] = value
            else:
                sub_init_params_dict = None

            if isinstance(node, (ComponentChoice, EstimatorComponent)):
                node.set_hyperparameters(configuration=sub_configuration.get_dictionary(),
                                         init_params=sub_init_params_dict)
            else:
                raise NotImplementedError('Not supported yet!')

        return self

    def get_hyperparameter_search_space(self, dataset_properties=None) -> ConfigurationSpace:
        cs = ConfigurationSpace()
        for name, step in self.steps_.items():
            step_configuration_space = step.get_hyperparameter_search_space(dataset_properties)
            cs.add_configuration_space(name, step_configuration_space)
        return cs


class SubPipeline(EstimatorComponent):

    def __init__(self, sub_wfs: List[Structure],
                 dataset_properties: dict = None):
        self.dataset_properties = dataset_properties
        self.pipelines: Dict[str, FlexiblePipeline] = {}

        for idx, wf in enumerate(sub_wfs):
            self.pipelines['pipeline_{}'.format(idx)] = FlexiblePipeline(wf, dataset_properties=self.dataset_properties)

    def fit(self, X, y=None, **fit_params):
        for node_name, pipeline in self.pipelines.items():
            pipeline.fit(X, y, **fit_params)
        return self

    def transform(self, X: np.ndarray):
        X_transformed = X
        for name, pipeline in self.pipelines.items():
            y_pred = pipeline.predict(X)
            X_transformed = np.hstack((X_transformed, np.reshape(y_pred, (-1, 1))))

        return X_transformed

    def set_hyperparameters(self, configuration: dict, init_params=None):
        for node_name, pipeline in self.pipelines.items():
            sub_config_dict = {}
            for param in configuration:
                if param.startswith('{}:'.format(node_name)):
                    value = configuration[param]
                    new_name = param.replace('{}:'.format(node_name), '', 1)
                    sub_config_dict[new_name] = value
            pipeline.set_hyperparameters(sub_config_dict, init_params)

    def get_hyperparameter_search_space(self, dataset_properties=None):
        cs = ConfigurationSpace()
        for pipeline_name, pipeline in self.pipelines.items():
            pipeline_cs = ConfigurationSpace()

            for task_name, task in pipeline.steps_.items():
                step_configuration_space = task.get_hyperparameter_search_space(dataset_properties)
                pipeline_cs.add_configuration_space(task_name, step_configuration_space)
            cs.add_configuration_space(pipeline_name, pipeline_cs)
        return cs

    @staticmethod
    def get_properties(dataset_properties=None):
        return {}
