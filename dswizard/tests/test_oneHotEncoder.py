from unittest import TestCase

import numpy as np
import numpy.testing as np_test

from dswizard.components.feature_preprocessing.one_hot_encoding import OneHotEncoder


class TestOneHotEncoder(TestCase):
    def test_numeric_only(self):
        # Generate 3 samples with 2 features
        X = np.zeros((3, 2))

        ohe = OneHotEncoder()
        actual = ohe.transform(X)

        np_test.assert_allclose(actual, X)

    def test_categorical_only(self):
        X = np.array([['London', 'car'],
                      ['Berlin', 'bus'],
                      ['New York', 'subway']])

        expected = np.array([[0, 1, 0, 0, 1, 0],
                             [1, 0, 0, 1, 0, 0],
                             [0, 0, 1, 0, 0, 1]])

        ohe = OneHotEncoder()
        actual = ohe.transform(X)

        np_test.assert_allclose(actual, expected)

    def test_mixed(self):
        X = np.array([['London', 'car', 20, 1.5],
                      ['Berlin', 'bus', 10, 1.6],
                      ['New York', 'subway', 30, 1.7]])
        expected = np.array([[0, 1, 0, 0, 1, 0, 20, 1.5],
                             [1, 0, 0, 1, 0, 0, 10, 1.6],
                             [0, 0, 1, 0, 0, 1, 30, 1.7]])

        ohe = OneHotEncoder()
        actual = ohe.transform(X)

        np_test.assert_allclose(actual, expected)
