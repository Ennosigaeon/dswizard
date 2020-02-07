import numpy as np
from sklearn.feature_selection import VarianceThreshold

from dswizard.components.feature_preprocessing.variance_threshold import VarianceThresholdComponent
from dswizard.tests.components import base_test


class TestVarianceThreshold(base_test.BaseComponentTest):

    def test_default(self):
        X_train, X_test, y_train, y_test = self.load_data()

        actual = VarianceThresholdComponent()
        actual.fit(X_train, y_train)
        X_actual = actual.transform(X_test)

        expected = VarianceThreshold()
        expected.fit(X_train, y_train)
        X_expected = expected.transform(X_test)

        assert np.allclose(X_actual, X_expected)
        assert repr(actual.preprocessor) == repr(expected)

    def test_configured(self):
        X_train, X_test, y_train, y_test = self.load_data()

        actual = VarianceThresholdComponent()
        config: dict = self.get_config(actual)

        actual.set_hyperparameters(config)
        actual.fit(X_train, y_train)
        X_actual = actual.transform(np.copy(X_test))

        expected = VarianceThreshold(**config)
        expected.fit(X_train, y_train)
        X_expected = expected.transform(X_test)

        assert np.allclose(X_actual, X_expected)
        assert repr(actual.preprocessor) == repr(expected)
