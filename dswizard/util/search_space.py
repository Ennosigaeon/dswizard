import inspect
from typing import List

import networkx as nx

from dswizard.components import classification, data_preprocessing, feature_preprocessing
from dswizard.components.base import EstimatorComponent, TunablePredictor, TunableEstimator
from dswizard.components.classification import ClassifierChoice
from dswizard.components.data_preprocessing import DataPreprocessorChoice
from dswizard.components.feature_preprocessing import FeaturePreprocessorChoice
from dswizard.pipeline.pipeline import FlexiblePipeline, SubPipeline
from dswizard.util import util


class StructureSearchSpace:

    # noinspection PyProtectedMember
    def __init__(self,
                 include_basic_estimators: bool = False):

        self.G: nx.DiGraph = nx.DiGraph()
        self.nodes: List[int] = []
        self.idx: int = 0

        # TODO SubPipelines missing
        self.candidates = {
            ClassifierChoice.name(),
            DataPreprocessorChoice.name(),
            FeaturePreprocessorChoice.name(),
            # SubPipeline.name()
        }

        if include_basic_estimators:
            for estimator in classification._classifiers.values():
                self.candidates.add(estimator.name())
            for estimator in data_preprocessing._preprocessors.values():
                self.candidates.add(estimator.name())
            for estimator in feature_preprocessing._preprocessors.values():
                self.candidates.add(estimator.name())

    def get_pipeline(self, node: int, expand_subpipelines: bool = False) -> FlexiblePipeline:
        # Remove artificial root node
        steps = []
        for idx in nx.shortest_path(self.G, 0, node)[1:]:
            steps.append((str(idx), self._get_estimator_instance(self.G.nodes[idx]['estimator'])))
        return FlexiblePipeline(steps)

    def build(self, max_depth: int = 5, max_split: int = 5) -> nx.DiGraph:
        self.G = nx.DiGraph()
        self.nodes = []
        self.idx = 0

        self.G.add_node(self.idx, label='root', estimator=None, valid=False, **self._get_style(False))
        self.nodes.append(self.idx)
        self.idx += 1

        # Depth-first expansion of search space
        while len(self.nodes) != 0:
            node = self.nodes.pop()

            depth = nx.shortest_path_length(self.G, 0, node)
            if depth < max_depth:
                self._expand_node(node)

        # Trim invalid leaf nodes
        # noinspection PyCallingNonCallable
        for node in [x for x in self.G.nodes() if self.G.out_degree(x) == 0 and self.G.in_degree(x) == 1]:
            if not self.G.nodes[node]['valid']:
                self.G.remove_node(node)

        valid_nodes = [n for n in self.G.nodes if self.G.nodes[n]['valid']]
        print(valid_nodes)
        print(len(valid_nodes))

        return self.G

    def _expand_node(self, node: int):
        for candidate in self.candidates:
            self.G.add_node(self.idx, label=candidate.split('.')[-1].replace('Choice', ''), estimator=candidate)
            self.G.add_edge(node, self.idx)

            valid = self._valid_pipeline(self.idx)
            self.G.add_node(self.idx, valid=valid, **self._get_style(valid))

            self.nodes.append(self.idx)
            self.idx += 1

    # noinspection PyMethodMayBeStatic
    def _get_label(self, name: str, short: bool = True) -> str:
        return name.split('.')[-1] if short else name

    def _valid_pipeline(self, node: int) -> bool:
        return self.G.nodes[node]['estimator'] == ClassifierChoice.name()

    # noinspection PyMethodMayBeStatic
    def _get_style(self, valid: bool) -> dict:
        if valid:
            return {'fillcolor': 'white', 'style': 'filled'}
        else:
            return {'fillcolor': 'red', 'style': 'filled'}

    # noinspection PyMethodMayBeStatic
    def _get_estimator_instance(self, clazz: str) -> EstimatorComponent:
        try:
            return util.get_object(clazz)
        except TypeError:
            if clazz == SubPipeline.name():
                return SubPipeline([])

            estimator = util.get_type(clazz)
            if 'predict' in inspect.getmembers(estimator, inspect.isfunction):
                # noinspection PyTypeChecker
                return TunablePredictor(estimator)
            else:
                # noinspection PyTypeChecker
                return TunableEstimator(estimator)


if __name__ == '__main__':
    search_space = StructureSearchSpace()
    G = search_space.build(max_depth=8)

    H = nx.nx_agraph.to_agraph(G)
    H.draw('search_space.pdf', prog='dot')
