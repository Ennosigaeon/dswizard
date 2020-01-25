import numpy as np
import pandas as pd
from ConfigSpace.configuration_space import ConfigurationSpace

from dswizard.components.base import PreprocessingAlgorithm


class OneHotEncoder(PreprocessingAlgorithm):
    def __init__(self,
                 categorical_features=None,
                 random_state=None):
        super().__init__()
        self.categorical_features = categorical_features

    def transform(self, X: np.ndarray):
        categorical = []
        numeric = []
        for i in range(X.shape[1]):
            try:
                X[:, i].astype(float)
                numeric.append(i)
            except ValueError:
                categorical.append(i)

        if len(categorical) == 0:
            return X

        df = pd.DataFrame.from_records(X[:, categorical])
        df = pd.get_dummies(df)

        for i in numeric:
            df[len(df.columns)] = X[:, i].astype(float)
        return df.to_numpy()

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': '1Hot',
                'name': 'One Hot Encoder',
                'handles_regression': True,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                # TODO find out of this is right!
                'handles_sparse': True,
                'handles_dense': True, }
        # 'input': (DENSE, SPARSE, UNSIGNED_DATA),
        # 'output': (INPUT,), }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        return cs
