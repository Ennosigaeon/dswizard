import numpy as np
import sklearn.naive_bayes

from dswizard.components.classification.gaussian_nb import GaussianNB
from dswizard.tests.components import base_test


class TestGaussianNB(base_test.BaseComponentTest):

    def test_default(self):
        X_train, X_test, y_train, y_test = self.load_data()

        actual = GaussianNB()
        actual.fit(X_train, y_train)
        y_actual = actual.predict(X_test)

        expected = sklearn.naive_bayes.GaussianNB()
        expected.fit(X_train, y_train)
        y_expected = expected.predict(X_test)

        assert np.allclose(y_actual, y_expected)
        assert repr(actual.estimator) == repr(expected)

    def test_configured(self):
        X_train, X_test, y_train, y_test = self.load_data()

        actual = GaussianNB()
        config: dict = self.get_config(actual)

        actual.set_hyperparameters(config)
        actual.fit(X_train, y_train)
        y_actual = actual.predict(X_test)

        expected = sklearn.naive_bayes.GaussianNB(**config)
        expected.fit(X_train, y_train)
        y_expected = expected.predict(X_test)

        assert np.allclose(y_actual, y_expected)
        assert repr(actual.estimator) == repr(expected)
