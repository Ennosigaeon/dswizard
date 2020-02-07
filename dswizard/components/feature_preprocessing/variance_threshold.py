from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import UniformFloatHyperparameter

from dswizard.components.base import PreprocessingAlgorithm


class VarianceThresholdComponent(PreprocessingAlgorithm):
    def __init__(self, threshold: float = 0.):
        super().__init__()
        self.threshold = threshold

    def fit(self, X, y=None):
        import sklearn.feature_selection
        self.preprocessor = sklearn.feature_selection.VarianceThreshold(threshold=self.threshold)
        self.preprocessor = self.preprocessor.fit(X)
        return self

    @staticmethod
    def get_properties(dataset_properties=None):
        return {
            'shortname': 'Variance Threshold',
            'name': 'Variance Threshold (constant feature removal)',
            'handles_regression': True,
            'handles_classification': True,
            'handles_multiclass': True,
            'handles_multilabel': True,
            'is_deterministic': True,
            'handles_sparse': True,
            'handles_dense': True,
            # 'input': (DENSE, SPARSE, UNSIGNED_DATA),
            # 'output': (INPUT,),
        }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        # TODO upper limit is totally ad hoc
        threshold = UniformFloatHyperparameter('threshold', 0., 1, default_value=0.)
        cs.add_hyperparameter(threshold)
        return cs
