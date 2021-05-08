import glob
import os
import pickle
from argparse import ArgumentParser
from typing import List, Union

import networkx as nx
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import FunctionTransformer
from tpot.builtins import StackingEstimator


def load_tpot(base_dir: str):
    def resolve_type(step):
        if isinstance(step, StackingEstimator):
            return type(step.estimator).__name__
        elif isinstance(step, FeatureUnion):
            ls = []
            for _, trans in step.transformer_list:
                if isinstance(trans, FunctionTransformer):
                    ls.append('Clone')
                else:
                    ls.append(resolve_type(trans))
            return ls
        else:
            return type(step).__name__

    # noinspection PyUnresolvedReferences
    def load_model(input: str):
        from sklearn.ensemble import AdaBoostClassifier
        from sklearn.naive_bayes import BernoulliNB
        from sklearn.tree import DecisionTreeClassifier

        from sklearn.ensemble._hist_gradient_boosting.gradient_boosting import HistGradientBoostingClassifier
        from sklearn.svm import SVC
        from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
        from sklearn.naive_bayes import MultinomialNB
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.linear_model import SGDClassifier

        from sklearn.preprocessing import MaxAbsScaler
        from sklearn.preprocessing import MinMaxScaler
        from sklearn.preprocessing import Normalizer
        from sklearn.preprocessing import QuantileTransformer
        from sklearn.preprocessing import RobustScaler
        from sklearn.preprocessing import StandardScaler

        from sklearn.neural_network import BernoulliRBM
        from sklearn.preprocessing import Binarizer
        from sklearn.decomposition import FactorAnalysis
        from sklearn.decomposition import FastICA
        from sklearn.cluster import FeatureAgglomeration
        from sklearn.feature_selection import GenericUnivariateSelect
        from sklearn.preprocessing import KBinsDiscretizer
        from sklearn.decomposition import KernelPCA
        from sklearn.impute import MissingIndicator
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import PolynomialFeatures
        from sklearn.ensemble import RandomTreesEmbedding
        from sklearn.feature_selection import SelectKBest
        from sklearn.feature_selection import SelectPercentile
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_selection import VarianceThreshold

        from sklearn.pipeline import FeatureUnion
        from sklearn.preprocessing import FunctionTransformer
        from tpot.builtins import OneHotEncoder
        from tpot.builtins import StackingEstimator
        import numpy as np

        try:
            pipeline: Pipeline = eval(input)['pipeline']
        except Exception as ex:
            print(file)
            print(input)
            pipeline = None
        return pipeline

    models = []
    for file in glob.glob(os.path.join(base_dir, 'tpot/**/models.txt'), recursive=True):
        with open(file, 'r') as f:
            content = ''.join(f.readlines())
            for e in content.split('}'):
                if not e.strip():
                    continue
                pipeline = load_model(e + '}')
                if pipeline is None:
                    continue

                steps = ['LabelEncoder', 'SimpleImputer']
                for _, step in pipeline.steps:
                    steps.append(resolve_type(step))
                models.append(steps)
    return models


def load_autosklearn(base_dir: str):
    rename = {
        'AdaboostClassifier': 'AdaBoostClassifier',
        'BinarizerComponent': 'Binarizer',
        'CategoricalImputation': 'SimpleImputer',
        'DecisionTree': 'DecisionTreeClassifier',
        'GenericUnivariateSelectComponent': 'GenericUnivariateSelect',
        'GradientBoostingClassifier': 'HistGradientBoostingClassifier',
        'KBinsDiscretizerComponent': 'KBinsDiscretizer',
        'LDA': 'LinearDiscriminantAnalysis',
        'LibSVM_SVC': 'SVC',
        'MinMaxScalerComponent': 'MinMaxScaler',
        'NormalizerComponent': 'Normalizer',
        'NumericalImputation': 'SimpleImputer',
        'QuantileTransformerComponent': 'QuantileTransformer',
        'RandomForest': 'RandomForestClassifier',
        'RobustScalerComponent': 'RobustScaler',
        'SGD': 'SGDClassifier',
        'StandardScalerComponent': 'StandardScaler',
        'SelectPercentileClassification': 'SelectPercentile'
    }

    from autosklearn.pipeline.base import AutoSklearnChoice
    from autosklearn.pipeline.components.data_preprocessing.data_preprocessing import DataPreprocessor

    def resolve_type(step):
        if isinstance(step, AutoSklearnChoice):
            name = type(step.estimator).__name__
            return rename[name] if name in rename else name
        elif isinstance(step, DataPreprocessor):
            ls = []
            for _, pipeline in step._transformers:
                tmp = []
                for _, s in pipeline.steps:
                    t = resolve_type(s)
                    if t not in {'CategoryShift', 'NoCoalescence', 'NoEncoding', 'NoRescalingComponent'}:
                        tmp.append(t)
                ls.append(tmp)
            return ls
        else:
            name = type(step).__name__
            return rename[name] if name in rename else name

    # noinspection PyUnresolvedReferences
    def load_model(input: str):
        from autosklearn.pipeline.classification import SimpleClassificationPipeline
        try:
            raw = [p.steps for _, p in eval(input)]
            models = []
            for pipeline in raw:
                steps = ['LabelEncoder']
                for _, step in pipeline:
                    n = resolve_type(step)
                    if isinstance(n, list):
                        steps.append(n)
                    elif n not in {'NoRescalingComponent'}:
                        steps.append(n)
                models.append(steps)
            return models
        except Exception as ex:
            print(ex, input)
            return []

    with open('fig/configspace.pkl', 'rb') as f:
        cs = pickle.load(f)

    from autosklearn.pipeline.classification import SimpleClassificationPipeline
    SimpleClassificationPipeline._get_hyperparameter_search_space = lambda *args, **kwargs: cs

    models = []
    for file in glob.glob(os.path.join(base_dir, 'autosklearn/**/models.txt'), recursive=True):
        with open(file, 'r') as f:
            content = ''.join(f.readlines())
            models.extend(load_model(content))
    return models


def load_dswizard(base_dir: str, variant: str):
    def load_model(ensemble):
        from dswizard.components.base import NoopComponent

        ls = []
        for pipeline in ensemble.estimators_:
            pip = []
            for _, est in pipeline.steps:
                est = est.estimator

                if isinstance(est, ColumnTransformer):
                    est = est.transformers[0][1]
                if isinstance(est, NoopComponent):
                    continue
                pip.append(type(est).__name__)
            ls.append(pip)
        return ls

    models = []
    for file in glob.glob(os.path.join(base_dir, variant, '**/models.pkl'), recursive=True):
        print(file)
        with open(file, 'rb') as f:
            try:
                tmp = pickle.load(f)
                try:
                    _, ensemble = tmp
                except ValueError:
                    _, ensemble, _ = tmp
                models.extend(load_model(ensemble))
            except EOFError as ex:
                print(ex)
    return models


def flatten(models: List[Union[str, List]]):
    flattened = []
    for pipeline in models:
        candidates = [[]]
        for step in pipeline:
            if isinstance(step, str):
                for candidate in candidates:
                    candidate.append(step)
            else:
                new_candidates = []
                for candidate in candidates:
                    for elem in step:
                        cp = candidate.copy()
                        if isinstance(elem, str):
                            cp.append(elem)
                        else:
                            cp.extend(elem)
                        new_candidates.append(cp)
                candidates = new_candidates
        flattened.extend(candidates)
    return flattened


replacements = {
    'AdaBoostClassifier': 'Classifier',
    'BernoulliNB': 'Classifier',
    'DecisionTreeClassifier': 'Classifier',
    'HistGradientBoostingClassifier': 'Classifier',
    'LinearDiscriminantAnalysis': 'Classifier',
    'SVC': 'Classifier',
    'MultinomialNB': 'Classifier',
    'RandomForestClassifier': 'Classifier',
    'SGDClassifier': 'Classifier',
    'SimpleImputer': 'Imputer',
    'KNNImputer': 'Imputer',
    'MaxAbsScaler': 'Scaling',
    'MinMaxScaler': 'Scaling',
    'Normalizer': 'Scaling',
    'RobustScaler': 'Scaling',
    'StandardScaler': 'Scaling',
    'QuantileTransformer': 'Scaling',
    'OneHotEncoder': 'Encoding',
    'LabelEncoder': 'Encoding',
    'FactorAnalysis': 'Decomposition',
    'FastICA': 'Decomposition',
    'FeatureAgglomeration': 'Decomposition',
    'KernelPCA': 'Decomposition',
    'PCA': 'Decomposition',
    'TruncatedSVD': 'Decomposition',
    'SelectKBest': 'Selection',
    'SelectPercentile': 'Selection',
    'GenericUnivariateSelect': 'Selection',
    'Binarizer': 'Discretization',
    'KBinsDiscretizer': 'Discretization',
    'PolynomialFeatures': 'Generation',
    'RandomTreesEmbedding': 'Generation',
    'BernoulliRBM': 'Generation',
    'MinorityCoalescer': 'Filter',
    'VarianceThreshold': 'Filter',
    'Clone': 'Generation'
}


def coalesce(models: List[Union[str, List[str]]]):
    coalesced = []
    for pipeline in models:
        coalesced.append([replacements.get(step, step) for step in pipeline])
    return coalesced


def to_prefixed_names(models: List[Union[str, List[str]]]):
    prefixed = []
    for pipeline in models:
        prefixed.append(['__ROOT'] + [f'{idx}_{name}' for idx, name in enumerate(pipeline)])
    return prefixed


def build_graph(models: List[List[str]], name: str, prune_factor: float = 0.025) -> nx.Graph:
    G = nx.DiGraph()
    for pipeline in models:
        for i in range(len(pipeline) - 1):
            data = G.get_edge_data(pipeline[i], pipeline[i + 1])
            if data is None:
                G.add_edge(pipeline[i], pipeline[i + 1], weight=1.0)
            else:
                data['weight'] = data['weight'] + 1.0

    max_weight = len(models)
    for u, v in G.edges():
        weight = G[u][v]['weight'] / max_weight
        G[u][v]['weight'] = weight
        G[u][v]['label'] = f'{G[u][v]["weight"]:.4f}'
        G[u][v]['fontsize'] = 5
        G[u][v]['penwidth'] = max(0.25, 6 * weight)

    for n, data in G.nodes(data=True):
        # data['label'] = n[2:]
        weight = sum([G[p][n]['weight'] for p in G.predecessors(n)])
        if weight < prune_factor:
            G.nodes[n]['fillcolor'] = 'gray'
            G.nodes[n]['style'] = 'filled'

    H = nx.nx_agraph.to_agraph(G)
    H.draw(f'fig/{name}_full.pdf', prog='dot')

    min_weight = 3 / len(models)
    to_remove = []
    for u, v in G.edges():
        if G[u][v]['weight'] < min_weight:
            to_remove.append((u, v))
    G.remove_edges_from(to_remove)
    # G.remove_nodes_from(list(nx.isolates(G)))

    for u, v in G.edges():
        G[u][v]['label'] = f'{G[u][v]["weight"]:.4f}'
        G[u][v]['fontsize'] = 5
        G[u][v]['penwidth'] = max(0.25, 6 * G[u][v]['weight'])

    H = nx.nx_agraph.to_agraph(G)
    H.draw(f'fig/{name}_pruned.pdf', prog='dot')

    return G


def build_circo_graph(models: List[List[str]], name: str, start: int = 0, prune_factor: float = 0.01,
                      normalize: bool = True) -> nx.Graph:
    G = nx.DiGraph()
    total_edges = 0
    total_nodes = 0

    print(name, np.mean(np.array([len(p) - start for p in models])), np.std(np.array([len(p) - start for p in models])))

    for pipeline in models:
        for i in range(start, len(pipeline)):
            total_nodes += 1
            try:
                G.nodes[pipeline[i]]['weight'] += 1
            except KeyError:
                G.add_node(pipeline[i], weight=1)

    for pipeline in models:
        if start == len(pipeline) - 1:
            total_edges += 1

        for i in range(start, len(pipeline) - 1):
            total_edges += 1
            data = G.get_edge_data(pipeline[i], pipeline[i + 1])
            if data is None:
                G.add_edge(pipeline[i], pipeline[i + 1], weight=1.0)
            else:
                data['weight'] = data['weight'] + 1.0

    def scale(x: float, borders=(0.01, 0.8), offset: float = 0.15):
        return (1 - offset) * (x - borders[0]) / (borders[1] - borders[0]) + offset if normalize else x

    for u, v in G.edges():
        G[u][v]['weight'] /= total_edges
        G[u][v]['label'] = f'{scale(G[u][v]["weight"], borders=(0.01, 0.25)):.4f}'
    for n in G.nodes():
        G.nodes[n]['weight'] /= total_nodes

    to_remove = []
    for u, v in G.edges():
        if G[u][v]['weight'] < prune_factor:
            to_remove.append((u, v))
    G.remove_edges_from(to_remove)

    for n in G.nodes():
        weight = G.nodes[n]['weight']
        G.nodes[n]['label'] = f'{n}: {weight:.4f}'
        if weight < prune_factor:
            G.nodes[n]['fillcolor'] = 'gray'
            G.nodes[n]['style'] = 'filled'

    H = nx.nx_agraph.to_agraph(G)
    H.draw(f'fig/{name}_circo.pdf', prog='circo')

    return G


parser = ArgumentParser()
parser.add_argument('base_dir', type=str, help='Base dir containing raw results')
parser.add_argument('--load', type=bool, help='Load raw results instead of cache', default=False)
args = parser.parse_args()

base_dir = args.base_dir
if args.load:
    autosklearn = load_autosklearn(base_dir)
    tpot = load_tpot(base_dir)
    dswizard = load_dswizard(base_dir, 'dswizard')
    dswizard_star = load_dswizard(base_dir, 'dswizard_star')
    with open('fig/models.pkl', 'wb') as f:
        pickle.dump((autosklearn, dswizard, dswizard_star, tpot), f)
else:
    with open('fig/models.pkl', 'rb') as f:
        autosklearn, dswizard, dswizard_star, tpot = pickle.load(f)

tpot = flatten(tpot)
tpot = coalesce(tpot)
build_circo_graph(tpot, 'tpot', start=2)
tpot = to_prefixed_names(tpot)
build_graph(tpot, 'tpot')
print(tpot)

autosklearn = flatten(autosklearn)
autosklearn = coalesce(autosklearn)
build_circo_graph(autosklearn, 'autosklearn', start=1)
autosklearn = to_prefixed_names(autosklearn)
build_graph(autosklearn, 'autosklearn')
print(autosklearn)

dswizard = flatten(dswizard)
dswizard = coalesce(dswizard)
build_circo_graph(dswizard, 'dswizard')
dswizard = to_prefixed_names(dswizard)
build_graph(dswizard, 'dswizard')
print(dswizard)

dswizard_star = flatten(dswizard_star)
dswizard_star = coalesce(dswizard_star)
build_circo_graph(dswizard_star, 'dswizard_star')
dswizard_star = to_prefixed_names(dswizard_star)
build_graph(dswizard_star, 'dswizard_star')
print(dswizard_star)
