import numpy as np
import sklearn.tree

from dswizard.components.classification.decision_tree import DecisionTree
from dswizard.tests.components import base_test


class TestDecisionTree(base_test.BaseComponentTest):

    def test_default(self):
        X_train, X_test, y_train, y_test = self.load_data()

        actual = DecisionTree(random_state=42)
        actual.fit(X_train, y_train)
        y_actual = actual.predict(X_test)

        expected = sklearn.tree.DecisionTreeClassifier(random_state=42)
        expected.fit(X_train, y_train)
        y_expected = expected.predict(X_test)

        assert np.allclose(y_actual, y_expected)
        # assert repr(actual.estimator) == repr(expected)

    def test_configured(self):
        X_train, X_test, y_train, y_test = self.load_data()

        actual = DecisionTree(random_state=42)
        config = self.get_config(actual)

        actual.set_hyperparameters(config)
        actual.fit(X_train, y_train)
        y_actual = actual.predict(X_test)

        config['max_depth'] = int(np.round(config['max_depth_factor'] * X_train.shape[1], 0))
        del config['max_depth_factor']

        expected = sklearn.tree.DecisionTreeClassifier(**config, random_state=42)
        expected.fit(X_train, y_train)
        y_expected = expected.predict(X_test)

        assert np.allclose(y_actual, y_expected)
        # assert repr(actual.estimator) == repr(expected)
