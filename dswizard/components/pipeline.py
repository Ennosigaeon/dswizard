from typing import Dict, List, Tuple, Union

import numpy as np
from ConfigSpace.configuration_space import Configuration, ConfigurationSpace, OrderedDict
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline

from dswizard.components.base import ComponentChoice, EstimatorComponent
from dswizard.util import util


class FlexiblePipeline(Pipeline, BaseEstimator):

    def __init__(self, steps: Dict[str, EstimatorComponent], dataset_properties: dict):
        super().__init__(list(steps.items()))
        self.steps_ = steps
        self.configuration = None
        self.dataset_properties = dataset_properties

        self.configuration_space: ConfigurationSpace = self.get_hyperparameter_search_space()

    def _validate_steps(self):
        if len(self.steps) == 0:
            raise TypeError('Pipeline has to contain at least 1 step')
        super()._validate_steps()
        if not hasattr(self.steps[-1][1], 'predict'):
            raise TypeError('Last step of Pipeline should implement predict.')

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
        if dataset_properties is None:
            dataset_properties = self.dataset_properties

        cs = ConfigurationSpace()
        for name, step in self.steps:
            step_configuration_space = step.get_hyperparameter_search_space(dataset_properties)
            cs.add_configuration_space(name, step_configuration_space)
        return cs

    def items(self):
        return self.steps_.items()

    def as_list(self) -> Tuple[List[Tuple[str, Union[str, List]]], Dict]:
        steps = []
        for name, step in self.steps:
            steps.append((name, step.serialize()))
        return steps, self.dataset_properties

    @staticmethod
    def from_list(steps: List[Tuple[str, Union[str, List]]], ds_properties: Dict) -> 'FlexiblePipeline':
        def __load(sub_steps: List[Tuple[str, Union[str, List]]]) -> Dict[str, EstimatorComponent]:
            d = OrderedDict()
            for name, value in sub_steps:
                if type(value) == str:
                    # TODO kwargs for __init__ not loaded
                    d[name] = util.get_object(value)
                elif type(value) == list:
                    ls = []
                    for sub_name, sub_value in value:
                        ls.append(__load(sub_value))
                    d[name] = SubPipeline(ls, ds_properties)
                else:
                    raise ValueError('Unable to handle type {}'.format(type(value)))
            return d

        ds = __load(steps)
        return FlexiblePipeline(ds, ds_properties)

    def __lt__(self, other: 'FlexiblePipeline'):
        s1 = tuple(e.name() for e in self.steps_.values())
        s2 = tuple(e.name() for e in other.steps_.values())
        return s1 < s2


class SubPipeline(EstimatorComponent):

    def __init__(self, sub_wfs: List[Dict[str, EstimatorComponent]],
                 dataset_properties: dict = None):
        self.dataset_properties = dataset_properties
        self.pipelines: Dict[str, FlexiblePipeline] = {}

        ls = list(map(lambda wf: FlexiblePipeline(wf, dataset_properties=self.dataset_properties), sub_wfs))
        for idx, wf in enumerate(sorted(ls)):
            self.pipelines['pipeline_{}'.format(idx)] = wf

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

            for task_name, task in pipeline.steps:
                step_configuration_space = task.get_hyperparameter_search_space(dataset_properties)
                pipeline_cs.add_configuration_space(task_name, step_configuration_space)
            cs.add_configuration_space(pipeline_name, pipeline_cs)
        return cs

    def serialize(self):
        pipelines = []
        for name, p in self.pipelines.items():
            pipelines.append((name, p.as_list()[0]))

        return pipelines

    @staticmethod
    def get_properties(dataset_properties=None):
        return {}
