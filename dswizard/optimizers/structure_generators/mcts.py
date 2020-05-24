import random
from abc import ABC
from copy import deepcopy
from typing import List, Optional, Set, Tuple, Type

import math
import networkx as nx
import numpy as np
from sklearn.base import is_classifier

from automl.components.base import EstimatorComponent
from automl.components.classification import ClassifierChoice
from automl.components.data_preprocessing import DataPreprocessorChoice
from automl.components.feature_preprocessing import FeaturePreprocessorChoice
from core.worker import Worker
from dswizard.components.pipeline import FlexiblePipeline
from dswizard.core.base_structure_generator import BaseStructureGenerator
from dswizard.core.model import CandidateStructure, Dataset
from dswizard.core.model import Result


# Ideas
#   1. Bandit Learning
#       - Iterate all pipelines with maximum depth d
#       - Use MCTS to select next pipeline to tune
#           - Use grammar to generate valid pipelines during simulation and expansion
#           - Consider pipeline complexity to avoid over-fitting
#           - Stop node selection to re-select a shorter pipeline
#           - Prior for step performance via OpenML flows
#           - Do not select next action indepently but consider previous steps. Algorithms are often selected in paris
#               (see MLPlan)
#       - Optimize hyperparameters via BO
#           - Prior for hyperparameters via OpenML flows
#
#  General:
#   - Multi-target optimization
#       - Quality
#       - Complexity
#       - Training time
#       - Hardware limitations
#


class Node(ABC):
    """
    A representation of a single board state.
    MCTS works by constructing a tree of these Nodes.
    Could be e.g. a chess or checkers board state.
    """

    def __init__(self,
                 id: int,
                 ds: Optional[Dataset],
                 estimator: Optional[Type[EstimatorComponent]],
                 pipeline_prefix: List[Tuple[str, EstimatorComponent]] = None
                 ):
        self.id = id
        self.ds = ds

        if estimator is None:
            self.label = 'ROOT'
            self.estimator = None
        else:
            self.estimator = estimator()
            self.label = self.estimator.name()

        if pipeline_prefix is None:
            pipeline_prefix = []
        else:
            pipeline_prefix = deepcopy(pipeline_prefix)
            # TODO check if also add if pipeline_prefix is None
            pipeline_prefix.append((str(id), self.estimator))
        self.steps: List[Tuple[str, EstimatorComponent]] = pipeline_prefix

        self.visits = 0
        self.reward = 0

    def is_terminal(self):
        return is_classifier(self.estimator)

    # noinspection PyMethodMayBeStatic
    def all_available_actions(self) -> Set[Type[EstimatorComponent]]:
        components = set()
        components.update(ClassifierChoice().get_components().values())
        components.update(DataPreprocessorChoice().get_components().values())
        components.update(FeaturePreprocessorChoice().get_components().values())
        return components

    def __hash__(self) -> int:
        return self.id

    def __eq__(self, node2: int) -> bool:
        return self.id == node2


class Tree:
    ROOT: int = 0

    def __init__(self, ds: Dataset):
        self.G = nx.DiGraph()
        self.id_count = -1
        self.add_node(None, ds)

        self.coef_progressive_widening = 0.6020599913279623

    def get_new_id(self) -> int:
        self.id_count += 1
        return self.id_count

    def add_node(self,
                 estimator: Optional[Type[EstimatorComponent]],
                 ds: Optional[Dataset],
                 parent_node: Node = None) -> Node:
        new_id = self.get_new_id()
        node = Node(new_id, ds, estimator, pipeline_prefix=parent_node.steps if parent_node is not None else None)

        self.G.add_node(new_id, value=node, label=node.label)
        if parent_node is not None:
            self.G.add_edge(parent_node, new_id)
        return node

    def get_node(self, node: int) -> Node:
        return self.G.nodes[node]['value']

    def get_children(self, node: int) -> List[Node]:
        return [self.G.nodes[n]['value'] for n in self.G.successors(node)]

    def get_all_children(self, node: int) -> List[Node]:
        return [self.G.nodes[n]['value'] for n in nx.dfs_tree(self.G, node).nodes.keys()]

    def get_predecessors(self, node: int) -> Node:
        return self.get_node(next(self.G.predecessors(node)))

    def fully_expanded(self, node: Node, global_max: int = 20):
        # global_max is used as failsafe to prevent massive expansion of single node

        possible_children = len(node.all_available_actions())
        max_children = math.floor(math.pow(node.visits, self.coef_progressive_widening))
        max_children = min(max_children, possible_children)

        current_children = len(list(self.G.successors(node.id)))

        return (current_children != 0 or possible_children == 0) and current_children >= min(global_max, max_children)

    def highlight_path(self, end: int, start: int = 0):
        for n in nx.shortest_path(self.G, source=start, target=end):
            self.G.nodes[n]['fillcolor'] = 'gray'
            self.G.nodes[n]['style'] = 'filled'

    def __contains__(self, node: int) -> bool:
        return node in self.G.nodes


class Policy:

    # TODO include meta-learning here

    def __init__(self, exploration_weight: float = 2):
        self.exploration_weight = exploration_weight

    # noinspection PyMethodMayBeStatic
    def get_next_action(self, n: Node, current_children: List[Node]) -> Optional[Type[EstimatorComponent]]:
        actions = n.all_available_actions()
        exhausted_actions = [n.estimator for n in current_children]
        actions = [a for a in actions if a not in exhausted_actions]

        if len(actions) == 0:
            return None
        return random.choice(actions)

    def uct(self, node: Node, tree: Tree) -> Node:
        """Select a child of node, balancing exploration & exploitation"""
        if node.visits == 0:
            log_N_vertex = 0
        else:
            log_N_vertex = math.log(node.visits)

        def uct(n: Node):
            """Upper confidence bound for trees"""
            if n.visits == 0:
                return math.inf

            return n.reward / n.visits + self.exploration_weight * math.sqrt(
                log_N_vertex / node.visits
            )

        return max(tree.get_children(node.id), key=uct)


class MCTS(BaseStructureGenerator):
    """Monte Carlo tree searcher. First rollout the tree then choose a move."""

    def __init__(self, worker: Worker, **kwargs):
        super().__init__(**kwargs)
        self.worker = worker
        self.tree: Optional[Tree] = None
        self.policy = Policy()

    def get_candidate(self, ds: Dataset) -> CandidateStructure:
        # Initialize tree if not exists
        if self.tree is None:
            self.tree = Tree(ds)

        # traverse from root to a leaf node
        self.logger.debug('MCTS SELECT')
        path = self._select()

        # expand last node in path if possible
        self.logger.debug('MCTS EXPAND')
        expansion = self._expand(path[-1])
        if expansion is not None:
            path.append(expansion)

        self.logger.debug('MCTS SIMULATE')
        for i in range(10):
            node = self._simulate(path)
            if node is not None:
                break
        else:
            raise ValueError(
                'Failed to obtain a valid pipeline structure during simulation for pipeline prefix [{}]'.format(
                    ', '.join([n.label for n in path])))

        print(node.steps)
        pipeline = FlexiblePipeline(node.steps)

        return CandidateStructure(pipeline.configuration_space, pipeline)

    def _select(self) -> List[Node]:
        """Find an unexplored descendent of `node`"""

        path: List[Node] = []
        node = self.tree.get_node(Tree.ROOT)
        while True:
            self.logger.debug('\tSelecting {}'.format(node.label))
            path.append(node)

            if not self.tree.fully_expanded(node):
                return path

            # node is terminal
            if len(self.tree.get_children(node.id)) == 0:
                return path

            node = self.policy.uct(node, self.tree)  # descend a layer deeper

    def _expand(self, node: Node) -> Optional[Node]:
        if self.tree.fully_expanded(node):
            return None

        n_actions = len(node.all_available_actions())
        n_children = len(self.tree.get_children(node.id))
        while n_children < n_actions:
            current_children = self.tree.get_children(node.id)
            action = self.policy.get_next_action(node, current_children)
            self.logger.debug('\tExpanding with {}. Option {}/{}'.format(action().name(), n_children + 1, n_actions))

            try:
                component = action()
                self.logger.debug('\tFitting with default ')
                # TODO calculate score for classifiers and pass to config generators
                X = self.worker.transform_dataset(node.ds,
                                                  component.get_hyperparameter_search_space().get_default_configuration(),
                                                  component)
                ds = Dataset(X, node.ds.y)
                # TODO validate that transformation was really successful

                return self.tree.add_node(estimator=action, ds=ds, parent_node=node)
            except KeyboardInterrupt:
                raise
            except Exception as ex:
                self.logger.debug('\t{} failed with {}'.format(action().name(), ex))
                new_child = self.tree.add_node(estimator=action, ds=None, parent_node=node)
                self._backpropagate([key for key, values in new_child.steps], np.inf)

                n_children += 1

    # noinspection PyMethodMayBeStatic
    def _simulate(self, path: List[Node], max_depths: int = 3) -> Optional[Node]:
        """Returns the reward for a random simulation (to completion) of `node`"""
        node = path[-1]
        for i in range(max_depths):
            if node.is_terminal():
                return node

            # TODO use meta-learning
            # TODO always select classifier as last step
            action = self.policy.get_next_action(node, [])
            if action is None:
                break

            node = Node(id=-(i + 1), ds=None, estimator=action, pipeline_prefix=node.steps)

        self.logger.warn('Failed to simulate pipeline with maximal depth {}'.format(max_depths))
        return None

    def register_result(self, candidate: CandidateStructure, result: Result, update_model: bool = True,
                        **kwargs) -> None:
        reward = result.loss

        if reward is None or not np.isfinite(reward):
            # One could skip crashed results, but we decided to assign a +inf loss and count them as bad configurations
            # Same for non numeric losses. Note that this means losses of minus infinity will count as bad!
            reward = np.inf

        self._backpropagate(candidate.pipeline.steps_.keys(), reward)

    # noinspection PyMethodMayBeStatic
    def _backpropagate(self, path: List[str], reward: float) -> None:
        """Send the reward back up to the ancestors of the leaf"""
        # Also update root node

        ls = list(path)
        ls.append('0')
        for name in ls:
            id = int(name)
            if id >= 0:
                n = self.tree.get_node(id)
                n.visits += 1
                n.reward += reward

    def choose(self, node: Node = None):
        """Choose the best successor of node. (Choose a move in the game)"""
        if node is None:
            node = self.tree.get_node(Tree.ROOT)

        children = self.tree.get_all_children(node.id)
        if len(children) == 0:
            return node

        def score(n: Node):
            if n.visits == 0 or not n.is_terminal():
                return math.inf  # avoid unseen moves
            return n.reward / n.visits  # average reward

        return min(children, key=score)
