from unittest import TestCase

from sklearn import datasets
from sklearn.model_selection import train_test_split

from dswizard.util.common import check_none


class BaseComponentTest(TestCase):

    def test_default(self):
        pass

    def test_configured(self):
        pass

    def load_data(self):
        X, y = datasets.load_iris(True)
        return train_test_split(X, y, test_size=0.33, random_state=42)

    def get_config(self, actual) -> dict:
        config: dict = actual.get_hyperparameter_search_space().sample_configuration().get_dictionary()
        for key, value in config.items():
            if check_none(value):
                config[key] = None
        print(config)
        return config
