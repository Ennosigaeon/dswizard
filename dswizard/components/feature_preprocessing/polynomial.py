from ConfigSpace.configuration_space import ConfigurationSpace
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformIntegerHyperparameter

from dswizard.components.base import PreprocessingAlgorithm
from dswizard.util.common import check_for_bool


class PolynomialFeaturesComponent(PreprocessingAlgorithm):
    def __init__(self, degree: int = 2, interaction_only: bool = False, include_bias: bool = True):
        super().__init__()
        self.degree = degree
        self.interaction_only = interaction_only
        self.include_bias = include_bias

    def fit(self, X, y=None):
        import sklearn.preprocessing

        self.interaction_only = check_for_bool(self.interaction_only)
        self.include_bias = check_for_bool(self.include_bias)

        self.preprocessor = sklearn.preprocessing.PolynomialFeatures(degree=self.degree,
                                                                     interaction_only=self.interaction_only,
                                                                     include_bias=self.include_bias)
        self.preprocessor.fit(X, y)
        return self

    @staticmethod
    def get_properties(dataset_properties=None):
        return {'shortname': 'PolynomialFeatures',
                'name': 'PolynomialFeatures',
                'handles_regression': True,
                'handles_classification': True,
                'handles_multiclass': True,
                'handles_multilabel': True,
                'is_deterministic': True,
                # 'input': (DENSE, UNSIGNED_DATA),
                # 'output': (INPUT,)
                }

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None):
        # More than degree 3 is too expensive!
        degree = UniformIntegerHyperparameter("degree", 2, 3, 2)
        interaction_only = CategoricalHyperparameter("interaction_only", [False, True], False)
        include_bias = CategoricalHyperparameter("include_bias", [True, False], True)

        cs = ConfigurationSpace()
        cs.add_hyperparameters([degree, interaction_only, include_bias])

        return cs
