import importlib
import inspect
import pkgutil
import sys
from abc import ABC
from collections import OrderedDict
from typing import Type, Dict, Any, Optional

import numpy as np
from ConfigSpace import ConfigurationSpace
from sklearn.base import BaseEstimator
from sklearn.utils import check_random_state, check_array


def find_components(package: str, directory: str, base_class: Type) -> Dict[str, Type]:
    components = OrderedDict()

    for module_loader, module_name, ispkg in pkgutil.iter_modules([directory]):
        full_module_name = "%s.%s" % (package, module_name)
        if full_module_name not in sys.modules and not ispkg:
            module = importlib.import_module(full_module_name)

            for member_name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, base_class) and \
                        obj != base_class:
                    # TODO test if the obj implements the interface
                    # Keep in mind that this only instantiates the ensemble_wrapper,
                    # but not the real target classifier
                    classifier = obj
                    components[module_name] = classifier

    return components


class MetaData:
    @staticmethod
    def get_properties(dataset_properties=None) -> dict:
        """Get the properties of the underlying algorithm.

        Find more information at :ref:`get_properties`

        Parameters
        ----------

        dataset_properties : dict, optional (default=None)

        Returns
        -------
        dict
        """
        raise NotImplementedError()

    @staticmethod
    def get_hyperparameter_search_space(dataset_properties=None) -> ConfigurationSpace:
        """Return the configuration space of this classification algorithm.

        Parameters
        ----------

        dataset_properties : dict, optional (default=None)

        Returns
        -------
        Configspace.configuration_space.ConfigurationSpace
            The configuration space of this classification algorithm.
        """
        raise NotImplementedError()

    def set_hyperparameters(self, configuration, init_params=None) -> 'MetaData':
        raise NotImplementedError()

    @classmethod
    def name(cls) -> str:
        return '.'.join([cls.__module__, cls.__qualname__])


# noinspection PyPep8Naming
class PredictionMixin:

    def predict(self, X: np.ndarray) -> np.ndarray:
        """The predict function calls the predict function of the
        underlying scikit-learn model and returns an array with the predictions.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        Returns
        -------
        array, shape = (n_samples,) or shape = (n_samples, n_labels)
            Returns the predicted values

        Notes
        -----
        Please see the `scikit-learn API documentation
        <http://scikit-learn.org/dev/developers/index.html#apis-of-scikit
        -learn-objects>`_ for further information."""
        raise NotImplementedError()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict probabilities.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        Returns
        -------
        array, shape=(n_samples,) if n_classes == 2 else (n_samples, n_classes)
        """
        raise NotImplementedError()


# noinspection PyPep8Naming
class EstimatorComponent(BaseEstimator, MetaData, ABC):

    def fit(self, X: np.ndarray, y: np.ndarray) -> 'EstimatorComponent':
        """The fit function calls the fit function of the underlying
        scikit-learn model and returns `self`.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)
            Training data

        y : array-like, shape = (n_samples,) or shape = (n_sample, n_labels)

        Returns
        -------
        self : returns an instance of self.
            Targets

        Notes
        -----
        Please see the `scikit-learn API documentation
        <http://scikit-learn.org/dev/developers/index.html#apis-of-scikit
        -learn-objects>`_ for further information."""
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """The transform function calls the transform function of the
        underlying scikit-learn model and returns the transformed array.

        Parameters
        ----------
        X : array-like, shape = (n_samples, n_features)

        Returns
        -------
        X : array
            Return the transformed training data

        Notes
        -----
        Please see the `scikit-learn API documentation
        <http://scikit-learn.org/dev/developers/index.html#apis-of-scikit
        -learn-objects>`_ for further information."""
        raise NotImplementedError()

    def serialize(self):
        # TODO kwargs for __init__ not persisted
        cls = self.__class__
        return '.'.join([cls.__module__, cls.__qualname__])

    def set_hyperparameters(self, configuration: dict, init_params=None) -> 'EstimatorComponent':
        for param, value in configuration.items():
            if not hasattr(self, param):
                raise ValueError('Cannot set hyperparameter %s for %s because '
                                 'the hyperparameter does not exist.' %
                                 (param, str(self)))
            setattr(self, param, value)

        if init_params is not None:
            for param, value in init_params.items():
                if not hasattr(self, param):
                    raise ValueError('Cannot set init param %s for %s because '
                                     'the init param does not exist.' %
                                     (param, str(self)))
                setattr(self, param, value)

        return self

    def __str__(self):
        cls = self.__class__
        return '.'.join([cls.__module__, cls.__qualname__])


# noinspection PyPep8Naming
class PredictionAlgorithm(EstimatorComponent, PredictionMixin, ABC):
    """Provide an abstract interface for classification algorithms in
    auto-sklearn.

    See :ref:`extending` for more information."""

    def __init__(self):
        self.estimator: Optional[BaseEstimator] = None
        self.properties: Optional[Dict] = None

    def get_estimator(self):
        """Return the underlying estimator object.

        Returns
        -------
        estimator : the underlying estimator object
        """
        return self.estimator

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = check_array(X)
        # add class probabilities as a synthetic feature
        # noinspection PyUnresolvedReferences
        X_transformed = np.hstack((X, self.estimator.predict_proba(X)))

        # add class prediction as a synthetic feature
        # noinspection PyUnresolvedReferences
        X_transformed = np.hstack((X_transformed, np.reshape(self.estimator.predict(X), (-1, 1))))

        return X_transformed


# noinspection PyPep8Naming
class PreprocessingAlgorithm(EstimatorComponent, ABC):
    """Provide an abstract interface for preprocessing algorithms in auto-sklearn.

    See :ref:`extending` for more information."""

    def __init__(self):
        self.preprocessor: Optional[BaseEstimator] = None

    def get_preprocessor(self) -> BaseEstimator:
        """Return the underlying preprocessor object.

        Returns
        -------
        preprocessor : the underlying preprocessor object
        """
        return self.preprocessor

    def fit_transform(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        return self.fit(X, y).transform(X)


# noinspection PyPep8Naming
class ComponentChoice(EstimatorComponent):

    def __init__(self, random_state=None):
        # Since all calls to get_hyperparameter_search_space will be done by the
        # pipeline on construction, it is not necessary to construct a
        # configuration space at this location!
        # self.configuration = self.get_hyperparameter_search_space(
        #     dataset_properties).get_default_configuration()

        if random_state is None:
            self.random_state = check_random_state(1)
        else:
            self.random_state = check_random_state(random_state)

        # Since the pipeline will initialize the hyperparameters, it is not
        # necessary to do this upon the construction of this object
        # self.set_hyperparameters(self.configuration)
        self.choice: Optional[BaseEstimator] = None
        self.configuration_space_: Optional[ConfigurationSpace] = None
        self.dataset_properties_: Optional[Dict] = None
        self.new_params = None

    def get_components(self) -> Dict[str, Any]:
        raise NotImplementedError()

    def get_available_components(self, dataset_properties: dict = None,
                                 include=None,
                                 exclude=None) -> Dict[str, Any]:
        if dataset_properties is None:
            dataset_properties = {}

        if include is not None and exclude is not None:
            raise ValueError(
                "The argument include and exclude cannot be used together.")

        available_comp = self.get_components()

        if include is not None:
            for incl in include:
                if incl not in available_comp:
                    raise ValueError("Trying to include unknown component: "
                                     "%s" % incl)

        components_dict = OrderedDict()
        for name in available_comp:
            if include is not None and name not in include:
                continue
            elif exclude is not None and name in exclude:
                continue

            # TODO maybe check for sparse?

            components_dict[name] = available_comp[name]

        return components_dict

    def set_hyperparameters(self, configuration: dict, init_params=None) -> 'ComponentChoice':
        new_params = {}

        choice = configuration['__choice__']
        del configuration['__choice__']

        for param, value in configuration.items():
            param = param.replace(choice, '').replace(':', '')
            new_params[param] = value

        if init_params is not None:
            for param, value in init_params.items():
                param = param.replace(choice, '').replace(':', '')
                new_params[param] = value

        new_params['random_state'] = self.random_state

        self.new_params = new_params
        self.choice = self.get_components()[choice](**new_params)

        return self

    def get_hyperparameter_search_space(self, dataset_properties: dict = None,
                                        default=None,
                                        include=None,
                                        exclude=None) -> ConfigurationSpace:
        raise NotImplementedError()

    def transform(self, X: np.ndarray) -> np.ndarray:
        # noinspection PyUnresolvedReferences
        return self.choice.transform(X)

    @staticmethod
    def get_properties(dataset_properties: dict = None) -> Dict:
        return {}


# noinspection PyPep8Naming
class TunableEstimator(EstimatorComponent):

    def __init__(self, estimator: Type[EstimatorComponent]):
        self.type: Type[EstimatorComponent] = estimator
        self.instance: Optional[EstimatorComponent] = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> EstimatorComponent:
        return self.instance.fit(X, y)

    def transform(self, X: np.ndarray) -> np.ndarray:
        return self.instance.transform(X)

    def fit_transform(self, X: np.ndarray, y: np.ndarray = None) -> np.ndarray:
        if hasattr(self.instance, 'fit_transform'):
            return self.instance.fit_transform(X, y)
        else:
            return self.instance.fit(X, y).transform(X)

    @staticmethod
    def get_properties(dataset_properties=None) -> Dict:
        pass

    def get_hyperparameter_search_space(self, dataset_properties=None) -> ConfigurationSpace:
        return self.type.get_hyperparameter_search_space(dataset_properties)

    def set_hyperparameters(self, configuration: dict, init_params=None) -> None:
        # noinspection PyArgumentList
        self.instance = self.type(**configuration)


# noinspection PyPep8Naming
class TunablePredictor(TunableEstimator, PredictionMixin):

    def __init__(self, estimator: Type[PredictionAlgorithm]):
        super().__init__(estimator)

        self.type: Type[PredictionAlgorithm] = estimator
        self.instance: Optional[PredictionAlgorithm] = None

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.instance.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.instance.predict_proba(X)
