import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace

from dswizard.components.base import PredictionAlgorithm


class GaussianNB(PredictionAlgorithm):

    def __init__(self, random_state=None, verbose: int = 0):
        super().__init__()
        self.random_state = random_state
        self.verbose = int(verbose)
        self.estimator = None
        self.classes_ = None

    def fit(self, X, y):
        import sklearn.naive_bayes

        self.estimator = sklearn.naive_bayes.GaussianNB()
        self.classes_ = np.unique(y.astype(int))

        # Fallback for multilabel classification
        if len(y.shape) > 1 and y.shape[1] > 1:
            import sklearn.multiclass
            self.estimator = sklearn.multiclass.OneVsRestClassifier(self.estimator, n_jobs=1)
        self.estimator.fit(X, y)

        return self

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'GaussianNB',
                'name': 'Gaussian Naive Bayes classifier',
                'handles_regression': False,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                # 'input': (DENSE, UNSIGNED_DATA),
                # 'output': (PREDICTIONS,)
                }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        cs = ConfigurationSpace()
        return cs
