import abc
import logging
import os
import pickle
import random
import threading
from abc import ABC
from copy import deepcopy
from typing import List, Optional, Tuple, Type, Dict

import joblib
import math
import networkx as nx
import numpy as np
from sklearn.base import is_classifier
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline

from automl.components.base import EstimatorComponent
from automl.components.classification import ClassifierChoice
from automl.components.data_preprocessing import DataPreprocessorChoice
from automl.components.feature_preprocessing import FeaturePreprocessorChoice
from automl.components.meta_features import MetaFeatures
from dswizard.components.pipeline import FlexiblePipeline
from dswizard.core.base_structure_generator import BaseStructureGenerator
from dswizard.core.model import CandidateId, PartialConfig, StatusType, CandidateStructure, Dataset, Result, \
    EvaluationJob
from dswizard.core.worker import Worker
from dswizard.util import util


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


class Node:
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
    def available_actions(self,
                          include_preprocessing: bool = True,
                          include_classifier: bool = True,
                          ignore_mf: bool = True) -> Dict[str, Type[EstimatorComponent]]:
        components = {}
        mf = self.ds.mf_dict if self.ds and not ignore_mf is not None else None
        if include_classifier:
            components.update(ClassifierChoice().get_available_components(mf=mf))
        if include_preprocessing:
            components.update(DataPreprocessorChoice().get_available_components(mf=mf))
            components.update(FeaturePreprocessorChoice().get_available_components(mf=mf))
        return components

    def enter(self):
        self.reward += util.worst_score(self.ds.metric)

    def exit(self):
        self.reward -= util.worst_score(self.ds.metric)

    def update(self, reward: float):
        self.visits += 1
        self.reward += reward

    def __hash__(self) -> int:
        return self.id

    def __eq__(self, node2: int) -> bool:
        return self.id == node2


class Tree:
    ROOT: int = 0

    def __init__(self, ds: Dataset):
        self.G = nx.DiGraph()
        self.id_count = -1
        self.add_node(ds=ds)

        self.coef_progressive_widening = 0.7
        self.lock = threading.Lock()

    def add_node(self,
                 estimator: Optional[Type[EstimatorComponent]] = None,
                 ds: Optional[Dataset] = None,
                 parent_node: Node = None) -> Node:
        self.id_count += 1
        new_id = self.id_count
        node = Node(new_id, ds, estimator, pipeline_prefix=parent_node.steps if parent_node is not None else None)
        if ds is None:
            node.failure_message = 'Incomplete'

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

    def fully_expanded(self, node: Node, global_max: int = 10):
        # global_max is used as failsafe to prevent massive expansion of single node

        possible_children = len(node.available_actions())
        max_children = math.floor(math.pow(node.visits, self.coef_progressive_widening))
        max_children = min(max_children, possible_children)

        current_children = len(list(self.G.successors(node.id)))

        return possible_children == 0 or (current_children != 0 and current_children >= min(global_max, max_children))

    def plot(self, file: str):
        # TODO plot can only be called once
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


class Policy(ABC):

    def __init__(self, logger: logging.Logger, exploration_weight: float = 2, **kwargs):
        self.logger = logger
        self.exploration_weight = exploration_weight

    @abc.abstractmethod
    def get_next_action(self,
                        n: Node,
                        current_children: List[Node],
                        include_preprocessing: bool = True,
                        include_classifier: bool = True,
                        depth: int = 1) -> Optional[Type[EstimatorComponent]]:
        pass

    def uct(self, n: Node, parent: Node, scale=2., force: bool = False) -> float:
        """Upper confidence bound for trees"""
        if n.visits == 0:
            return math.inf

        if n.failed:
            # Always ignore nodes without meta-features
            return -math.inf

        # Always pick terminal node if forced
        if force and n.is_terminal():
            return math.inf

        if parent is None or parent.visits == 0:
            log_N_vertex = 0
        else:
            log_N_vertex = math.log(parent.visits)

        exploitation = (-1 * n.reward) / n.visits  # HPO computes minimization problem. UCT selects maximum
        exploration = math.sqrt(log_N_vertex / n.visits)
        overfitting = (scale ** len(n.steps)) / (scale ** 10)
        return (exploitation + self.exploration_weight * exploration) * (1 - overfitting)

    def select(self, node: Node, tree: Tree) -> Tuple[Node, float]:
        """Select a child of node, balancing exploration & exploitation"""
        options = tree.get_children(node.id)
        scores = [self.uct(n, node) for n in options]
        idx = int(np.argmax(scores))
        return options[idx], scores[idx]


class RandomSelection(Policy):

    def get_next_action(self,
                        n: Node,
                        current_children: List[Node],
                        include_preprocessing: bool = True,
                        include_classifier: bool = True,
                        depth: int = 1) -> Optional[Type[EstimatorComponent]]:
        actions = n.available_actions(include_preprocessing=include_preprocessing,
                                      include_classifier=include_classifier,
                                      ignore_mf=depth > 1).values()
        exhausted_actions = [type(n.component) for n in current_children]
        actions = [a for a in actions if a not in exhausted_actions]

        if len(actions) == 0:
            return None
        return random.choice(actions)


class TransferLearning(Policy):

    def __init__(self, logger: logging.Logger, dir: str, model: str = None, task: int = None, **kwargs):
        super().__init__(logger, **kwargs)

        if model is None:
            model = os.path.join(dir, 'rf_d{}.pkl'.format(util.openml_mapping(task=task)))
        else:
            model = os.path.join(dir, model)

        logger.info('Loading transfer model from {}'.format(model))
        with open(model, 'rb') as f:
            mean, var = joblib.load(f)
        self.mean: Pipeline = mean
        self.var: Pipeline = var

    def get_next_action(self,
                        n: Node,
                        current_children: List[Node],
                        include_preprocessing: bool = True,
                        include_classifier: bool = True,
                        depth: int = 1) -> Optional[Type[EstimatorComponent]]:
        available_actions = n.available_actions(include_preprocessing=include_preprocessing,
                                                include_classifier=include_classifier,
                                                ignore_mf=depth > 1)
        exhausted_actions = [type(n.component) for n in current_children]
        actions = [key for key, value in available_actions.items() if value not in exhausted_actions]
        actions = np.atleast_2d(actions)

        step = np.atleast_2d(np.repeat(np.ones(1) * depth, actions.shape[1]))
        X = np.repeat(n.ds.meta_features, actions.shape[1], axis=0)
        X = np.hstack((X, actions.T, step.T))
        # remove land-marking mf
        X = np.delete(X, slice(42, 50), axis=1)

        mean = self.mean.predict(X)
        var = self.var.predict(X)
        perf = np.random.multivariate_normal(mean, np.diag(var))

        return available_actions[actions[0, np.argmax(perf)]]


class MCTS(BaseStructureGenerator):
    """Monte Carlo tree searcher. First rollout the tree then choose a move."""

    def __init__(self, cutoff: int, workdir: str, policy: Type[Policy] = None,
                 policy_kwargs: dict = None, store_ds: bool = False, **kwargs):
        super().__init__(**kwargs)
        self.workdir = workdir
        self.cutoff = cutoff
        self.tree: Optional[Tree] = None
        self.neighbours = NearestNeighbors()
        self.mfs: Optional[MetaFeatures] = None
        self.store_ds = store_ds

        if policy is None:
            policy = RandomSelection
        if policy_kwargs is None:
            policy_kwargs = {}
        try:
            self.policy = policy(self.logger, **policy_kwargs)
        except (KeyError, FileNotFoundError) as ex:
            self.logger.warning('Failed to initialize Policy: {}. Fallback to RandomSelection.'.format(ex))
            self.policy = RandomSelection(self.logger)

    def fill_candidate(self, cs: CandidateStructure, ds: Dataset, worker: Worker = None,
                       retries=3) -> CandidateStructure:
        # Initialize tree if not exists
        if self.tree is None:
            self.tree = Tree(ds)
            self.mfs = ds.meta_features
            self.neighbours.fit(self.mfs)

        # traverse from root to a leaf node
        self.logger.debug('MCTS SELECT')
        path, expand = self._select(force=retries == 0)

        result = None
        if expand or not path[-1].is_terminal():
            # expand last node in path if possible
            self.logger.debug('MCTS EXPAND')
            max_depths = 3
            max_mf_failures = 3
            for i in range(1, max_depths + 1):
                expansion, result, mf_failures = self._expand(path, worker, cs.cid, max_mf_missing=max_mf_failures,
                                                              include_preprocessing=i < max_depths)
                max_mf_failures -= mf_failures
                if expansion is not None:
                    path.append(expansion)
                    if expansion.is_terminal():
                        break
                else:
                    break
        else:
            self.logger.debug('Skipping MCTS expansion')

        if len(path) == 1:
            with self.tree.lock:
                for node in path:
                    self.tree.get_node(node.id).exit()
            self.logger.warning(
                'Current path contains only ROOT node. Trying tree traversal {} more times...'.format(retries))
            return self.fill_candidate(cs, ds, worker=worker, retries=retries - 1)
        if not path[-1].is_terminal():
            with self.tree.lock:
                for node in path:
                    self.tree.get_node(node.id).exit()
            self.logger.warning(
                'Current path does not end in a classifier. Trying tree traversal {} more times...'.format(retries))
            return self.fill_candidate(cs, ds, worker=worker, retries=retries - 1)

        # A simulation is not necessary. Simulated results are already incorporated in the policy

        node = path[-1]
        self.logger.debug('Sampled pipeline structure: {}'.format(node.steps))
        pipeline = FlexiblePipeline(node.steps)

        cs.configspace = pipeline.configuration_space
        cs.pipeline = pipeline
        cs.cfg_keys = [n.partial_config.cfg_key for n in path if n.partial_config is not None]

        # If no simulation was necessary, add the default configuration as first result
        if result is not None and result.loss is not None and result.loss < 1:
            cs.add_result(result)
        return cs

    def _select(self, force: bool = False) -> Tuple[List[Node], bool]:
        """Find an unexplored descendent of ROOT"""

        path: List[Node] = []
        node = self.tree.get_node(Tree.ROOT)
        score = -math.inf
        while True:
            self.logger.debug('\tSelecting {}'.format(node.label))
            if node.failed:
                self.logger.warning(
                    'Selected node has no dataset, meta-features or partial configurations. This should not happen')

            with self.tree.lock:
                node.enter()
                path.append(node)

                # Failsafe mechanism to enforce structure selection
                if force and node.is_terminal():
                    return path, False

                if not self.tree.fully_expanded(node):
                    return path, True

                # node is leaf
                if len(self.tree.get_children(node.id)) == 0:
                    return path, True

                candidate, candidate_score = self.policy.select(node, self.tree)  # descend a layer deeper
                # Current node is better than children or selected node failed
                if candidate.failed or candidate_score <= score:
                    return path, not node.is_terminal()
                else:
                    node, score = candidate, candidate_score

    def _expand(self, nodes: List[Node],
                worker: Worker, cid: CandidateId,
                max_distance: float = 1,
                max_mf_missing: int = 3,
                include_preprocessing: bool = True) -> Tuple[Optional[Node], Optional[Result], int]:
        node = nodes[-1]

        mf_missing_count = 0
        n_actions = len(node.available_actions())
        while True:
            with self.tree.lock:
                if node.is_terminal() and self.tree.fully_expanded(node):
                    return None, None, mf_missing_count

                n_children = len(self.tree.get_children(node.id))
                if n_children >= n_actions:
                    break

                current_children = self.tree.get_children(node.id)
                action = self.policy.get_next_action(node, current_children,
                                                     include_preprocessing=include_preprocessing)
                if action is None:
                    return None, None, mf_missing_count
                if mf_missing_count >= max_mf_missing:
                    self.logger.warning('Aborting expansion due to {} failed MF calculations'.format(mf_missing_count))
                    return None, None, mf_missing_count

                component = action()

                self.logger.debug(
                    '\tExpanding with {}. Option {}/{}'.format(component.name(), n_children + 1, n_actions))
                new_node = self.tree.add_node(estimator=action, parent_node=node)

            ds = node.ds
            config, key = self.cfg_cache.sample_configuration(
                configspace=component.get_hyperparameter_search_space(),
                mf=ds.meta_features,
                default=True)

            job = EvaluationJob(ds, cid.without_config(), cs=component, cutoff=self.cutoff, config=config,
                                cfg_keys=[key])
            result = worker.start_transform_dataset(job)

            if result.status.value == StatusType.SUCCESS.value:
                ds = Dataset(result.transformed_X, ds.y, ds.metric, ds.cutoff)
                new_node.partial_config = PartialConfig(key, config, str(new_node.id), ds.meta_features)
                new_node.ds = ds
                new_node.failure_message = None

                if ds.meta_features is None:
                    result.status = StatusType.CRASHED
                    result.loss = util.worst_score(ds.metric)
                    new_node.failure_message = 'Missing MF'
                    mf_missing_count += 1
                else:
                    # Check if any node in the tree is similar to the new dataset
                    distance, idx = self.neighbours.kneighbors(ds.meta_features, n_neighbors=1)
                    if np.allclose(node.ds.meta_features, ds.meta_features):
                        self.logger.debug('\t{} did not modify dataset'.format(component.name()))
                        result.status = StatusType.INEFFECTIVE
                        result.loss = util.worst_score(ds.metric)
                        new_node.failure_message = 'Ineffective'
                    elif distance[0][0] <= max_distance:
                        # TODO: currently always the existing node is selected. This node could represent simpler model
                        self.logger.debug('\t{} produced a dataset similar to {}'.format(component.name(), idx[0][0]))
                        result.status = StatusType.DUPLICATE
                        result.loss = util.worst_score(ds.metric)
                        new_node.failure_message = 'Duplicate {}'.format(idx[0][0])
                    else:
                        self.mfs = np.append(self.mfs, ds.meta_features, axis=0)
                        self.neighbours.fit(self.mfs)
                        node.enter()

                        if self.store_ds:
                            with open(os.path.join(self.workdir, '{}.pkl'.format(new_node.id)), 'wb') as f:
                                pickle.dump(ds, f)
            else:
                self.logger.debug(
                    '\t{} failed with as default hyperparamter: {}'.format(component.name(), result.status))
                result.loss = util.worst_score(ds.metric)
                new_node.failure_message = 'Crashed'

            if result.loss is not None:
                n_children += 1
                self._backpropagate([key for key, values in new_node.steps], result.loss)
                if result.loss < util.worst_score(ds.metric):
                    result.partial_configs = [n.partial_config for n in nodes if n.partial_config is not None]
                    result.partial_configs.append(new_node.partial_config)
                    result.config = FlexiblePipeline(new_node.steps).configuration_space.get_default_configuration()

                    job.result = result
                    self.cfg_cache.register_result(job)
                    # Successful classifiers
                    return new_node, result, mf_missing_count

            else:
                # Successful preprocessors
                return new_node, result, mf_missing_count
        return None, None, mf_missing_count

    def register_result(self, candidate: CandidateStructure, result: Result, update_model: bool = True,
                        **kwargs) -> None:
        reward = result.loss

        if reward is None or not np.isfinite(reward):
            reward = 1

        self._backpropagate(candidate.pipeline.steps_.keys(), reward, exit=True)

    # noinspection PyMethodMayBeStatic
    def _backpropagate(self, path: List[str], reward: float, exit: bool = False) -> None:
        """Send the reward back up to the ancestors of the leaf"""
        # Also update root node

        ls = list(path)
        ls.append('0')
        with self.tree.lock:
            for node_id in ls:
                node = self.tree.get_node(int(node_id))
                node.update(reward)
                if exit:
                    node.exit()

    def shutdown(self):
        if self.tree is None:
            self.logger.info("Search graph not initiated. Skipping rendering as pdf")
            return
        self.tree.plot(os.path.join(self.workdir, 'search_graph.pdf'))
