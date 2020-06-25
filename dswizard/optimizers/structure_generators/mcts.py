import random
from abc import ABC
from copy import deepcopy
from typing import List, Optional, Set, Tuple, Type

import math
import networkx as nx
import numpy as np
from sklearn.base import is_classifier
from sklearn.neighbors import NearestNeighbors

from automl.components.base import EstimatorComponent
from automl.components.classification import ClassifierChoice
from automl.components.data_preprocessing import DataPreprocessorChoice
from automl.components.feature_preprocessing import FeaturePreprocessorChoice
from automl.components.meta_features import MetaFeatures
from core.model import CandidateId, PartialConfig, StatusType
from core.worker import Worker
from dswizard.components.pipeline import FlexiblePipeline
from dswizard.core.base_structure_generator import BaseStructureGenerator
from dswizard.core.model import CandidateStructure, Dataset, Job
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
                 component: Optional[Type[EstimatorComponent]],
                 pipeline_prefix: List[Tuple[str, EstimatorComponent]] = None,
                 partial_config: PartialConfig = None
                 ):
        self.id = id
        self.ds = ds
        self.partial_config = partial_config

        self.failure_message: Optional[str] = None

        if component is None:
            self.label = 'ROOT'
            self.component = None
        else:
            self.component = component()
            self.label = self.component.name(short=True)

        if pipeline_prefix is None:
            pipeline_prefix = []
        else:
            pipeline_prefix = deepcopy(pipeline_prefix)
            # TODO check if also add if pipeline_prefix is None
            pipeline_prefix.append((str(id), self.component))
        self.steps: List[Tuple[str, EstimatorComponent]] = pipeline_prefix

        self.visits = 0
        self.reward = 0

    def is_terminal(self):
        return self.component is not None and is_classifier(self.component)

    @property
    def failed(self):
        return self.failure_message is not None

    # noinspection PyMethodMayBeStatic
    def available_actions(self, include_preprocessing: bool = True,
                          include_classifier: bool = True) -> Set[Type[EstimatorComponent]]:
        components = set()
        mf = self.ds.mf_dict if self.ds is not None else None
        if include_classifier:
            components.update(ClassifierChoice().get_available_components(mf=mf).values())
        if include_preprocessing:
            components.update(DataPreprocessorChoice().get_available_components(mf=mf).values())
            components.update(FeaturePreprocessorChoice().get_available_components(mf=mf).values())
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

    def fully_expanded(self, node: Node, global_max: int = 20):
        # global_max is used as failsafe to prevent massive expansion of single node

        possible_children = len(node.available_actions())
        max_children = math.floor(math.pow(node.visits, self.coef_progressive_widening))
        max_children = min(max_children, possible_children)

        current_children = len(list(self.G.successors(node.id)))

        return (current_children != 0 or possible_children == 0) and current_children >= min(global_max, max_children)

    def plot(self, file: str):
        for n, data in self.G.nodes(data=True):
            node: Node = data['value']
            if node.label.endswith('Component'):
                data['label'] = node.label[:-9]

            score = node.reward / node.visits if node.visits > 0 else 0
            data['label'] = '{} ({})\n{:.4f} / {}'.format(data['label'], n, score, node.visits)

            if node.failed:
                data['fillcolor'] = 'gray'
                data['style'] = 'filled'
                data['label'] += '\n' + node.failure_message

        H = nx.nx_agraph.to_agraph(self.G)
        H.draw(file, prog='dot')

    def __contains__(self, node: int) -> bool:
        return node in self.G.nodes


class Policy:

    # TODO include meta-learning here

    def __init__(self, exploration_weight: float = 2):
        self.exploration_weight = exploration_weight

    # noinspection PyMethodMayBeStatic
    def get_next_action(self,
                        n: Node,
                        current_children: List[Node],
                        include_preprocessing: bool = True,
                        include_classifier: bool = True) -> Optional[Type[EstimatorComponent]]:
        actions = n.available_actions(include_preprocessing=include_preprocessing,
                                      include_classifier=include_classifier)
        exhausted_actions = [type(n.component) for n in current_children]
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

            if n.ds.meta_features is None or n.partial_config is None:
                # Always ignore nodes without meta-features
                return 0

            return n.reward / n.visits + self.exploration_weight * math.sqrt(
                log_N_vertex / node.visits
            )

        return max(tree.get_children(node.id), key=uct)


class MCTS(BaseStructureGenerator):
    """Monte Carlo tree searcher. First rollout the tree then choose a move."""

    def __init__(self, worker: Worker, cutoff: int, **kwargs):
        super().__init__(**kwargs)
        self.worker = worker
        self.cutoff = cutoff
        self.tree: Optional[Tree] = None
        self.policy = Policy()
        self.neighbours = NearestNeighbors()
        self.mfs: Optional[MetaFeatures] = None

    def get_candidate(self, ds: Dataset) -> CandidateStructure:
        # Initialize tree if not exists
        if self.tree is None:
            self.tree = Tree(ds)
            self.mfs = ds.meta_features
            self.neighbours.fit(self.mfs)

        # traverse from root to a leaf node
        self.logger.debug('MCTS SELECT')
        path = self._select()

        # TODO add option to skip tree expansion
        # expand last node in path if possible
        self.logger.debug('MCTS EXPAND')
        expansion, result = self._expand(path)
        if expansion is not None:
            path.append(expansion)

        if len(path) == 1:
            # TODO selecting is deterministic. Retrying yields same result
            self.logger.warning('Current path contains only ROOT node. Trying tree traversal again...')
            return self.get_candidate(ds)

        self.logger.debug('MCTS SIMULATE')
        for i in range(10):
            node = self._simulate(path)
            if node is not None:
                break
        else:
            raise ValueError(
                'Failed to obtain a valid pipeline structure during simulation for pipeline prefix [{}]'.format(
                    ', '.join([n.label for n in path])))

        self.logger.debug('Sampled pipeline structure: {}'.format(node.steps))
        pipeline = FlexiblePipeline(node.steps)

        cs = CandidateStructure(pipeline.configuration_space, pipeline,
                                [n.partial_config.cfg_key for n in path if n.partial_config is not None])

        # If no simulation was necessary, add the default configuration as first result
        if result is not None and result.loss is not None and result.loss < 1:
            cs.add_result(result)
        return cs

    def _select(self) -> List[Node]:
        """Find an unexplored descendent of `node`"""

        path: List[Node] = []
        node = self.tree.get_node(Tree.ROOT)
        while True:
            self.logger.debug('\tSelecting {}'.format(node.label))
            if node.failed:
                self.logger.warning(
                    'Selected node has no dataset, meta-features or partial configurations. This should not happen')

            path.append(node)

            if not self.tree.fully_expanded(node):
                return path

            # node is leaf
            if len(self.tree.get_children(node.id)) == 0:
                return path

            # TODO guarantee that no failed algorithm nodes are selected
            node = self.policy.uct(node, self.tree)  # descend a layer deeper

    def _expand(self, nodes: List[Node], max_distance: float = 1) -> Tuple[Optional[Node], Optional[Result]]:
        node = nodes[-1]
        if self.tree.fully_expanded(node):
            return None, None

        n_actions = len(node.available_actions())
        n_children = len(self.tree.get_children(node.id))
        while n_children < n_actions:
            current_children = self.tree.get_children(node.id)
            action = self.policy.get_next_action(node, current_children)
            component = action()

            self.logger.debug('\tExpanding with {}. Option {}/{}'.format(component.name(), n_children + 1, n_actions))
            new_node = self.tree.add_node(estimator=action, ds=None, parent_node=node)

            ds = node.ds
            config, key = self.cfg_cache.sample_configuration(
                configspace=component.get_hyperparameter_search_space(),
                mf=ds.meta_features,
                default=True)

            job = Job(ds, CandidateId(-1, -1, -1), cs=component, cutoff=self.cutoff, config=config, cfg_keys=[key])
            result = self.worker.start_transform_dataset(job)

            if result.status.value == StatusType.SUCCESS.value:
                ds = Dataset(result.transformed_X, node.ds.y)

                if ds.meta_features is None:
                    result.status = StatusType.CRASHED
                    result.loss = 1
                    new_node.failure_message = 'Missing MF'
                else:
                    # Check if any node in the tree is similar to the new dataset
                    distance, idx = self.neighbours.kneighbors(ds.meta_features, n_neighbors=1)
                    if np.allclose(node.ds.meta_features, ds.meta_features):
                        self.logger.debug('\t{} did not modify dataset'.format(component.name()))
                        result.status = StatusType.INEFFECTIVE
                        result.loss = 1
                        new_node.failure_message = 'Ineffective'
                    elif distance[0][0] <= max_distance:
                        # TODO: currently always the existing node is selected. This node could represent simpler model
                        self.logger.debug('\t{} produced a dataset similar to {}'.format(component.name(), idx[0][0]))
                        result.status = StatusType.DUPLICATE
                        result.loss = 1
                        new_node.failure_message = 'Duplicate {}'.format(idx[0][0])
                    else:
                        self.mfs = np.append(self.mfs, ds.meta_features, axis=0)
                        self.neighbours.fit(self.mfs)

                partial_config = PartialConfig(key, config, str(node.id), ds.meta_features)
                new_node.partial_config = partial_config
                new_node.ds = ds
            else:
                self.logger.debug(
                    '\t{} failed with as default hyperparamter: {}'.format(component.name(), result.status))
                result.loss = 1
                new_node.failure_message = 'Crashed'

            if result.loss is not None:
                n_children += 1
                self._backpropagate([key for key, values in new_node.steps], result.loss)
                if result.loss < 1:
                    result.partial_configs = [n.partial_config for n in nodes if n.partial_config is not None]
                    result.partial_configs.append(new_node.partial_config)
                    result.config = PartialConfig.merge(result.partial_configs)

                    job.result = result
                    self.cfg_cache.register_result(job)
                    # Successful classifiers
                    return new_node, result

            else:
                # Successful preprocessors
                return new_node, result
        return None, None

    # noinspection PyMethodMayBeStatic
    def _simulate(self, path: List[Node], max_depths: int = 1) -> Optional[Node]:
        """Returns the reward for a random simulation (to completion) of `node`"""
        node = path[-1]
        for i in range(max_depths):
            if node.is_terminal():
                break

            # TODO use meta-learning
            action = self.policy.get_next_action(node, [], include_preprocessing=i < max_depths - 1)
            if action is None:
                break

            node = Node(id=-(i + 1), ds=None, component=action, pipeline_prefix=node.steps)

        if node.is_terminal():
            return node
        else:
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

    def shutdown(self):
        self.tree.plot('search_graph.pdf')
