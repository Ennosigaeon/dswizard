import logging
from abc import abstractmethod, ABC
from collections import defaultdict
from typing import List, Optional

import numpy as np
import pandas as pd
from pymfe.mfe import MFE

import pynisher2

LOGGER = logging.getLogger('mf')

MetaFeatures = np.ndarray


# ##########################################################################
# #  Extracting MetaFeatures with the help of AutoSklearn  #################
# ##########################################################################

class AbstractMetaFeature(ABC):

    @abstractmethod
    def calculate(self, X, y):
        pass


class NumberOfMissingValues(AbstractMetaFeature):
    def calculate(self, X, y):
        X_numeric = X.select_dtypes(include=['number'])
        X_object = X.select_dtypes(include=['category', 'object'])

        if X_object.empty:
            missing = ~np.isfinite(X_numeric)
            missing = missing.sum().sum()

            return int(missing)
        else:
            missing_o = pd.isna(X_object)
            missing_o = missing_o.sum().sum()

            missing_n = ~np.isfinite(X_numeric)
            missing_n = missing_n.sum().sum()

            missing = missing_n + missing_o

            return int(missing)


class PercentageOfMissingValues(AbstractMetaFeature):
    def calculate(self, X, y):
        X_numeric = X.select_dtypes(include=['number'])
        X_object = X.select_dtypes(include=['category', 'object'])

        if X_object.empty:
            missing = ~np.isfinite(X_numeric)
            missing = missing.sum().sum()

            return (float(missing) / float(X.shape[0] * X.shape[1])) * 100
        else:
            missing_o = pd.isna(X_object)
            missing_o = missing_o.sum().sum()

            missing_n = ~np.isfinite(X_numeric)
            missing_n = missing_n.sum().sum()

            missing = missing_n + missing_o

            return (float(missing) / float(X.shape[0] * X.shape[1])) * 100


class NumberOfInstancesWithMissingValues(AbstractMetaFeature):
    def calculate(self, X, y):
        X_numeric = X.select_dtypes(include=['number'])
        X_object = X.select_dtypes(include=['category', 'object'])

        if X_object.empty:
            missing = ~np.isfinite(X_numeric)
            num_missing = missing.sum(axis=1)

            return int(np.sum([1 if num > 0 else 0 for num in num_missing]))
        else:
            missing_o = pd.isna(X_object)
            num_missing_o = missing_o.sum(axis=1)

            missing_n = ~np.isfinite(X_numeric)
            num_missing_n = missing_n.sum(axis=1)
            num_missing = num_missing_n + num_missing_o

            return int(np.sum([1 if num > 0 else 0 for num in num_missing]))


class NumberOfFeaturesWithMissingValues(AbstractMetaFeature):
    def calculate(self, X, y):
        X_numeric = X.select_dtypes(include=['number'])
        X_object = X.select_dtypes(include=['category', 'object'])

        if X_object.empty:
            missing = ~np.isfinite(X_numeric)
            return int((missing.sum(axis=0) > 0).sum())
        else:
            missing_o = pd.isna(X_object)
            num_missing_o = (missing_o.sum(axis=0) > 0).sum()

            missing_n = ~np.isfinite(X_numeric)
            num_missing_n = (missing_n.sum(axis=0) > 0).sum()
            return int(num_missing_n + num_missing_o)


class ClassOccurrences(AbstractMetaFeature):
    def calculate(self, X, y):
        if len(y.shape) == 2:
            occurrences = []
            for i in range(y.shape[1]):
                occurrences.append(self.calculate(X, y[:, i]))
            return occurrences
        else:
            occurrence_dict = defaultdict(float)
            for value in y:
                occurrence_dict[value] += 1
            return occurrence_dict


class ClassProbabilityMean(AbstractMetaFeature):
    def calculate(self, X, y):
        occurrence_dict = ClassOccurrences().calculate(X, y)

        if len(y.shape) == 2:
            occurrences = []
            for i in range(y.shape[1]):
                occurrences.extend([occurrence for occurrence in occurrence_dict[i].value.values()])
            occurrences = np.array(occurrences)
        else:
            occurrences = np.array([occurrence for occurrence in occurrence_dict.values()], dtype=np.float64)
        return float((occurrences / y.shape[0]).mean())


class ClassProbabilitySTD(AbstractMetaFeature):
    def calculate(self, X, y):
        occurrence_dict = ClassOccurrences().calculate(X, y)

        if len(y.shape) == 2:
            stds = []
            for i in range(y.shape[1]):
                std = np.array([occurrence for occurrence in occurrence_dict[i].values()], dtype=np.float64)
                std = (std / y.shape[0]).std()
                stds.append(std)
            return np.mean(stds)
        else:
            occurrences = np.array([occurrence for occurrence in occurrence_dict.values()], dtype=np.float64)
            return float((occurrences / y.shape[0]).std())


class MetaFeatureFactory(object):

    @staticmethod
    def calculate(X: np.ndarray,
                  y: np.ndarray,
                  max_nan_percentage: float = 0.9,
                  max_features: int = 10000,
                  random_state: int = 42,
                  timeout: int = 600,
                  memory: int = 6144) -> Optional[MetaFeatures]:
        """
        Calculates the meta-features for the given DataFrame. The actual computation is dispatched to another process
        to prevent crashes due to extensive memory usage.
        :param X:
        :param y:
        :param max_nan_percentage:
        :param max_features:
        :param random_state:
        :param timeout:
        :param memory:
        :return:
        """
        wrapper = pynisher2.enforce_limits(wall_time_in_s=timeout, mem_in_mb=memory)(MetaFeatureFactory._calculate)
        res = wrapper(X, y, max_nan_percentage=max_nan_percentage, max_features=max_features,
                      random_state=random_state)
        if wrapper.exit_status is pynisher2.TimeoutException or wrapper.exit_status is pynisher2.MemorylimitException:
            return None
        elif wrapper.exit_status is pynisher2.AnythingException:
            LOGGER.warning('Failed to extract MF due to {}'.format(res[0]))
            return None
        elif wrapper.exit_status == 0 and res is not None:
            return np.array([res])

    @staticmethod
    def _calculate(X: np.ndarray,
                   y: np.ndarray,
                   max_nan_percentage: float = 0.9,
                   max_features: int = 10000,
                   random_state: int = 42) -> Optional[List[float]]:
        """
        Calculates the meta-features for the given DataFrame. _Attention_: Meta-feature calculation can require a lot of
        memory. This method should not be called directly to prevent the caller from crashing.
        :param X:
        :param y:
        :param max_nan_percentage:
        :param max_features:
        :param random_state:
        :return:
        """
        # Checks if number of features is bigger than max_features.
        if X.shape[1] > max_features:
            LOGGER.info('Number of features is bigger then {}'.format(max_features))
            return None

        X = pd.DataFrame(X, index=range(X.shape[0]), columns=range(X.shape[1]))

        # Extracting Missing Value Meta Features with AutoSklearn
        nr_missing_values = NumberOfMissingValues().calculate(X, y)
        pct_missing_values = PercentageOfMissingValues().calculate(X, y)
        nr_inst_mv = NumberOfInstancesWithMissingValues().calculate(X, y)
        nr_attr_mv = NumberOfFeaturesWithMissingValues().calculate(X, y)

        # Meta-Feature calculation does not work with missing data
        numeric = X.select_dtypes(include=['number']).columns
        if np.any(pd.isna(X)):
            n = X.shape[0]

            for i in X.columns:
                col = X[i]
                nan = pd.isna(col)
                if not nan.any():
                    continue
                elif nan.value_counts(normalize=True)[True] > max_nan_percentage:
                    X.drop(i, axis=1, inplace=True)
                elif i in numeric:
                    filler = np.random.normal(col.mean(), col.std(), n)
                    X[i] = col.combine_first(pd.Series(filler))
                else:
                    items = col.dropna().unique()
                    probability = col.value_counts(dropna=True, normalize=True)
                    probability = probability.where(probability > 0).dropna()
                    filler = np.random.choice(items, n, p=probability)
                    X[i] = col.combine_first(pd.Series(filler))

        for i in X.columns:
            col = X[i]
            if i in numeric:
                if not (abs(col - col.iloc[0]) > 1e-7).any() or (~np.isfinite(col)).all():
                    X.drop(i, inplace=True, axis=1)
            else:
                if not (col != col.iloc[0]).any():
                    X.drop(i, inplace=True, axis=1)

        if X.shape[0] == 0 or X.shape[1] == 0:
            LOGGER.info('X has no samples, no features or only constant values. Marking dataset as skipped.')
            return None
        """
       Selects Meta Features and extracts them
       """
        mfe = MFE(features=(['nr_inst', 'nr_attr', 'nr_class', 'nr_outliers', 'skewness', 'kurtosis', 'cor', 'cov',
                             'sparsity', 'var', 'class_ent', 'attr_ent', 'mut_inf',
                             'eq_num_attr', 'ns_ratio', 'nodes', 'leaves', 'leaves_branch', 'nodes_per_attr',
                             'var_importance', 'one_nn', 'best_node', 'linear_discr',
                             'naive_bayes', 'leaves_per_class']))
        mfe.fit(X.to_numpy(), y, transform_cat=True)
        f_name, f_value = mfe.extract(cat_cols='auto', suppress_warnings=True)

        """
        Mapping values to Meta Feature variables
        """
        nr_inst = int(f_value[f_name.index('nr_inst')])
        nr_attr = int(f_value[f_name.index('nr_attr')])
        nr_class = int(f_value[f_name.index('nr_class')])
        nr_outliers = int(f_value[f_name.index('nr_outliers')])
        class_ent = float(f_value[f_name.index('class_ent')])
        eq_num_attr = float(f_value[f_name.index('eq_num_attr')])
        ns_ratio = float(f_value[f_name.index('ns_ratio')])
        nodes = float(f_value[f_name.index('nodes')])
        leaves = float(f_value[f_name.index('leaves')])
        nodes_per_attr = float(f_value[f_name.index('nodes_per_attr')])

        def get_value(key: str):
            try:
                return float(f_value[f_name.index(key)])
            except:
                return float(f_value[f_name.index(key.split('.')[0])])

        skewness_mean = get_value('skewness.mean')
        skewness_sd = get_value('skewness.sd') if nr_attr > 1 else 0

        kurtosis_mean = get_value('kurtosis.mean')
        kurtosis_sd = get_value('kurtosis.sd') if nr_attr > 1 else 0

        cor_mean = get_value('cor.mean') if nr_attr > 1 else 1
        cor_sd = get_value('cor.sd') if nr_attr > 2 else 0

        cov_mean = get_value('cov.mean') if nr_attr > 1 else 0
        cov_sd = get_value('cov.sd') if nr_attr > 2 else 0

        sparsity_mean = get_value('sparsity.mean')
        sparsity_sd = get_value('sparsity.sd') if nr_attr > 1 else 0

        var_mean = get_value('var.mean')
        var_sd = get_value('var.sd') if nr_attr > 1 else 0

        attr_ent_mean = get_value('attr_ent.mean')
        attr_ent_sd = get_value('attr_ent.sd') if nr_attr > 1 else 0

        mut_inf_mean = get_value('mut_inf.mean')
        mut_inf_sd = get_value('mut_inf.sd') if nr_attr > 1 else 0

        leaves_branch_mean = get_value('leaves_branch.mean')
        leaves_branch_sd = get_value('leaves_branch.sd')

        leaves_per_class_mean = get_value('leaves_per_class.mean')
        leaves_per_class_sd = get_value('leaves_per_class.sd')
        # not sure under which conditions this exactly happens.
        if np.isnan(leaves_per_class_sd):
            leaves_per_class_sd = 0

        var_importance_mean = get_value('var_importance.mean')
        var_importance_sd = get_value('var_importance.sd') if nr_attr > 1 else 0

        one_nn_mean = get_value('one_nn.mean')
        one_nn_sd = get_value('one_nn.sd')

        best_node_mean = get_value('best_node.mean')
        best_node_sd = get_value('best_node.sd')

        linear_discr_mean = get_value('linear_discr.mean')
        linear_discr_sd = get_value('linear_discr.sd')

        naive_bayes_mean = get_value('naive_bayes.mean')
        naive_bayes_sd = get_value('naive_bayes.sd')

        # ##########################################################################
        # #  Extracting Meta Features with AutoSklearn  ############################
        # ##########################################################################

        pct_inst_mv = (float(nr_inst_mv) / float(nr_inst)) * 100

        pct_attr_mv = (float(nr_attr_mv) / float(nr_attr)) * 100

        class_prob_mean = ClassProbabilityMean().calculate(X, y)

        class_prob_std = ClassProbabilitySTD().calculate(X, y)

        # Meta-features must have exactly same order as in mlb
        # TODO normalize all values
        return [nr_inst, nr_attr, nr_class, nr_missing_values, pct_missing_values,
                nr_inst_mv, pct_inst_mv, nr_attr_mv, pct_attr_mv, nr_outliers,
                skewness_mean, skewness_sd, kurtosis_mean, kurtosis_sd, cor_mean, cor_sd,
                cov_mean, cov_sd, sparsity_mean, sparsity_sd, var_mean, var_sd,
                class_prob_mean, class_prob_std, class_ent, attr_ent_mean, attr_ent_sd,
                mut_inf_mean, mut_inf_sd, eq_num_attr, ns_ratio, nodes, leaves,
                leaves_branch_mean, leaves_branch_sd, nodes_per_attr, leaves_per_class_mean,
                leaves_per_class_sd, var_importance_mean, var_importance_sd, one_nn_mean,
                one_nn_sd, best_node_mean, best_node_sd, linear_discr_mean, linear_discr_sd,
                naive_bayes_mean, naive_bayes_sd]
