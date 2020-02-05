from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter

from dswizard.components.base import PreprocessingAlgorithm


class RobustScalerComponent(PreprocessingAlgorithm):
    def __init__(self, q_min: float = 25.0, q_max: float = 75.0):
        super().__init__()

        self.q_min = q_min
        self.q_max = q_max

    def fit(self, X, y=None):
        from sklearn.preprocessing import RobustScaler
        self.preprocessor = RobustScaler(quantile_range=(self.q_min, self.q_max), copy=False, )
        return super().fit(X, y)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'RobustScaler',
                'name': 'RobustScaler',
                'handles_regression': True,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                # TODO find out of this is right!
                'handles_sparse': True,
                'handles_dense': True,
                # 'input': (DENSE, UNSIGNED_DATA),
                # 'output': (INPUT, SIGNED_DATA),
                'preferred_dtype': None}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        q_min = UniformFloatHyperparameter('q_min', 0.001, 0.3, default_value=0.25)
        q_max = UniformFloatHyperparameter('q_max', 0.7, 0.999, default_value=0.75)
        cs.add_hyperparameters((q_min, q_max))
        return cs
