import numpy as np
from sklearn.ensemble._hist_gradient_boosting.gradient_boosting import HistGradientBoostingClassifier

from dswizard.components.classification.gradient_boosting import GradientBoostingClassifier
from dswizard.tests.components import base_test


class TestGradientBoosting(base_test.BaseComponentTest):

    def test_default(self):
        X_train, X_test, y_train, y_test = self.load_data()

        actual = GradientBoostingClassifier(random_state=42)
        actual.fit(X_train, y_train)
        y_actual = actual.predict(X_test)

        expected = HistGradientBoostingClassifier(random_state=42)
        expected.fit(X_train, y_train)
        y_expected = expected.predict(X_test)

        assert np.allclose(y_actual, y_expected)
        assert repr(actual.estimator) == repr(expected)

    def test_configured(self):
        X_train, X_test, y_train, y_test = self.load_data()

        actual = GradientBoostingClassifier(random_state=42)
        config: dict = self.get_config(actual)

        actual.set_hyperparameters(config)
        actual.fit(X_train, y_train)
        y_actual = actual.predict(X_test)

        expected = HistGradientBoostingClassifier(**config, random_state=42)
        expected.fit(X_train, y_train)
        y_expected = expected.predict(X_test)

        assert np.allclose(y_actual, y_expected)
        assert repr(actual.estimator) == repr(expected)
