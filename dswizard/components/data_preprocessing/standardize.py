from scipy import sparse

from dswizard.components.base import PreprocessingAlgorithm


class StandardScalerComponent(PreprocessingAlgorithm):
    def __init__(self):
        super().__init__()
        from sklearn.preprocessing import StandardScaler
        self.preprocessor = StandardScaler(copy=False)

    def fit(self, X, y=None):
        if sparse.isspmatrix(X):
            self.preprocessor.set_params(with_mean=False)

        return super(StandardScalerComponent, self).fit(X, y)

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'StandardScaler',
                'name': 'StandardScaler',
                'handles_missing_values': False,
                'handles_nominal_values': False,
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
                # 'input': (SPARSE, DENSE, UNSIGNED_DATA),
                # 'output': (INPUT,),
                'preferred_dtype': None}
