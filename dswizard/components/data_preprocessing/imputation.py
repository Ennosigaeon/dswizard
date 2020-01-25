from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter

from dswizard.components.base import PreprocessingAlgorithm


class Imputation(PreprocessingAlgorithm):
    def __init__(self,
                 strategy: str = 'median',
                 random_state=None):
        super().__init__()
        self.strategy = strategy

    def fit(self, X, y=None):
        import sklearn.impute

        self.preprocessor = sklearn.impute.SimpleImputer(
            strategy=self.strategy, copy=False)
        self.preprocessor = self.preprocessor.fit(X)
        return self

    def transform(self, X):
        if self.preprocessor is None:
            raise NotImplementedError()
        return self.preprocessor.transform(X)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'Imputation',
                'name': 'Imputation',
                'handles_missing_values': True,
                'handles_nominal_values': True,
                'handles_numerical_features': True,
                'prefers_data_scaled': False,
                'prefers_data_normalized': False,
                'handles_regression': True,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                # TODO find out of this is right!
                'handles_sparse': True,
                'handles_dense': True,
                # 'input': (DENSE, SPARSE, UNSIGNED_DATA),
                # 'output': (INPUT,),
                'preferred_dtype': None}

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        # TODO add replace by zero!
        strategy = CategoricalHyperparameter(
            "strategy", ["mean", "median", "most_frequent"], default_value="mean")
        cs = ConfigurationSpace()
        cs.add_hyperparameter(strategy)
        return cs
