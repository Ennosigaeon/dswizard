import numpy as np
from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, UniformFloatHyperparameter

from dswizard.components.base import PreprocessingAlgorithm
from dswizard.util.common import check_for_bool


class PCAComponent(PreprocessingAlgorithm):
    def __init__(self,
                 n_components: float = None,
                 whiten: bool = False,
                 random_state=None):
        super().__init__()
        self.n_components = n_components
        self.whiten = whiten

        self.random_state = random_state

    def fit(self, X, Y=None):
        import sklearn.decomposition
        self.whiten = check_for_bool(self.whiten)

        self.preprocessor = sklearn.decomposition.PCA(n_components=self.n_components,
                                                      whiten=self.whiten,
                                                      random_state=self.random_state,
                                                      copy=False)
        self.preprocessor.fit(X)

        if not np.isfinite(self.preprocessor.components_).all():
            raise ValueError("PCA found non-finite components.")

        return self

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'PCA',
                'name': 'Principle Component Analysis',
                'handles_regression': True,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                # TODO document that we have to be very careful
                'is_deterministic': False,
                # 'input': (DENSE, UNSIGNED_DATA),
                # 'output': (DENSE, UNSIGNED_DATA)
                }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        keep_variance = UniformFloatHyperparameter("n_components", 0.5, 0.9999, default_value=0.9999)
        whiten = CategoricalHyperparameter("whiten", [False, True], default_value=False)
        cs = ConfigurationSpace()
        cs.add_hyperparameters([keep_variance, whiten])
        return cs
