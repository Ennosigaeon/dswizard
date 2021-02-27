import abc
import logging
import math
import os
import pickle
import random
import threading
import timeit
from abc import ABC
from copy import deepcopy
from typing import List, Optional, Tuple, Type, Dict

import joblib
import networkx as nx
import numpy as np
from sklearn.base import is_classifier
from sklearn.pipeline import Pipeline

from dswizard.components.base import EstimatorComponent
from dswizard.components.classification import ClassifierChoice
from dswizard.components.data_preprocessing import DataPreprocessorChoice
from dswizard.components.feature_preprocessing import FeaturePreprocessorChoice
from dswizard.core.base_structure_generator import BaseStructureGenerator
from dswizard.core.model import CandidateId, PartialConfig, StatusType, CandidateStructure, Dataset, Result, \
    EvaluationJob
from dswizard.core.similaritystore import SimilarityStore
from dswizard.core.worker import Worker
from dswizard.pipeline.pipeline import FlexiblePipeline
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
                          include_classifier: bool = True) -> Dict[str, Type[EstimatorComponent]]:
        components = {}
        mf = self.ds.mf_dict if self.ds is not None else None
        if include_classifier:
            components.update(ClassifierChoice().get_available_components(mf=mf))
        if include_preprocessing:
            components.update(DataPreprocessorChoice().get_available_components(mf=mf))
            components.update(FeaturePreprocessorChoice().get_available_components(mf=mf))
        return components

    def enter(self):
        self.reward += util.worst_score(self.ds.metric)[-1]

    def exit(self):
        self.reward -= util.worst_score(self.ds.metric)[-1]

    def update(self, reward: float) -> 'Node':
        self.visits += 1
        self.reward += reward
        return self

    def __hash__(self) -> int:
        return self.id

    def __eq__(self, node2: int) -> bool:
        return self.id == node2

    @staticmethod
    def dummy(component, perf: float) -> 'Node':
        return Node(-1, None, component).update(perf)


class Tree:
    ROOT: int = 0

    def __init__(self, ds: Dataset):
        self.G = nx.DiGraph()
        self.id_count = -1
        self.add_node(ds=ds)

        self.coef_progressive_widening = 0.7

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
        max_children = min(max_children, possible_children, global_max)

        current_children = len(list(self.G.successors(node.id)))

        return possible_children == 0 or (current_children != 0 and current_children >= max_children)

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

    def __init__(self, logger: logging.Logger, exploration_weight: float = 2, wallclock_limit: float = None, **kwargs):
        self.logger = logger
        self.exploration_weight = exploration_weight
        self._exploration_weight = exploration_weight
        self.start = timeit.default_timer()
        self.wallclock_limit = wallclock_limit

    @abc.abstractmethod
    def get_next_action(self,
                        n: Node,
                        current_children: List[Node],
                        include_preprocessing: bool = True,
                        include_classifier: bool = True) -> Optional[Type[EstimatorComponent]]:
        pass

    def estimate_performance(self, actions: List[str], ds: Dataset, depth: int = 1):
        return util.worst_score(ds.metric)[-1] * np.ones(len(actions))

    def uct(self, n: Node, parent: Node, scale=2., force: bool = False) -> float:
        """Upper confidence bound for trees"""
        if n.failed or n.visits == 0:
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
        return (exploitation + self._exploration_weight * exploration) * (1 - overfitting)

    def select(self, node: Node, tree: Tree, force: bool = False) -> Tuple[Node, float]:
        """Select a child of node, balancing exploration & exploitation"""
        current_children = tree.get_children(node.id)
        if force:
            terminal_children = [child for child in current_children if child.is_terminal()]
            if len(terminal_children) > 0:
                children = terminal_children
            else:
                children = current_children
        elif not tree.fully_expanded(node):
            available_actions = node.available_actions()
            exhausted_actions = [type(n.component) for n in current_children]
            actions = [key for key, value in available_actions.items() if value not in exhausted_actions]
            perf = self.estimate_performance(actions, node.ds)

            possible_children = [Node.dummy(available_actions[c], p) for c, p in zip(actions, perf)]
            children = current_children + possible_children
        else:
            children = current_children

        if self.wallclock_limit is not None:
            self._exploration_weight = max(0.0, self.exploration_weight * (
                    math.exp((self.wallclock_limit + self.start - timeit.default_timer()) / self.wallclock_limit) - 1))

        scores = [self.uct(n, node) for n in children]
        idx = int(np.argmax(scores))

        # Selected non-existing child. Mark as "failed" to force abortion of select
        if idx >= len(current_children):
            children[idx].failure_message = 'Non Existing'

        return children[idx], scores[idx]


class RandomSelection(Policy):

    def get_next_action(self,
                        n: Node,
                        current_children: List[Node],
                        include_preprocessing: bool = True,
                        include_classifier: bool = True) -> Optional[Type[EstimatorComponent]]:
        actions = n.available_actions(include_preprocessing=include_preprocessing,
                                      include_classifier=include_classifier).values()
        exhausted_actions = [type(n.component) for n in current_children]
        actions = [a for a in actions if a not in exhausted_actions]

        if len(actions) == 0:
            return None
        return random.choice(actions)


class TransferLearning(Policy):

    def __init__(self, logger: logging.Logger, model: str, **kwargs):
        super().__init__(logger, **kwargs)

        logger.info('Loading transfer model from {}'.format(model))
        with open(model, 'rb') as f:
            mean, var = joblib.load(f)
        self.mean: Pipeline = mean
        self.var: Pipeline = var

    def estimate_performance(self, actions: List[str], ds: Dataset, depth: int = 1):
        if len(actions) == 0:
            return np.array([])
        actions = np.atleast_2d(actions)

        step = np.atleast_2d(np.repeat(np.ones(1) * depth, actions.shape[1]))
        X = np.repeat(ds.meta_features, actions.shape[1], axis=0)
        X = np.hstack((X, actions.T, step.T))
        # remove land-marking mf
        X = np.delete(X, slice(42, 50), axis=1)

        mean = self.mean.predict(X)
        var = self.var.predict(X)
        var = np.maximum(var, 0.01 * np.ones(var.shape))

        perf = np.random.multivariate_normal(mean, np.diag(var))
        return perf

    def get_next_action(self,
                        n: Node,
                        current_children: List[Node],
                        include_preprocessing: bool = True,
                        include_classifier: bool = True) -> Optional[Type[EstimatorComponent]]:
        available_actions = n.available_actions(include_preprocessing=include_preprocessing,
                                                include_classifier=include_classifier)
        exhausted_actions = [type(n.component) for n in current_children]
        actions = [key for key, value in available_actions.items() if value not in exhausted_actions]
        if len(actions) == 0:
            return None

        perf = self.estimate_performance(actions, n.ds)
        return available_actions[actions[int(np.argmax(perf))]]


class MCTS(BaseStructureGenerator):
    """Monte Carlo tree searcher. First rollout the tree then choose a move."""

    def __init__(self,
                 cutoff: int,
                 workdir: str,
                 policy: Type[Policy] = None,
                 store_ds: bool = False,
                 model: str = None,
                 wallclock_limit: float = None,
                 epsilon_greedy: bool = True,
                 **kwargs):
        super().__init__(**kwargs)
        self.workdir = workdir
        self.cutoff = cutoff
        self.tree: Optional[Tree] = None
        self.store_ds = store_ds
        self.lock = threading.Lock()

        if policy is None:
            policy = RandomSelection
        policy_kwargs = {
            'model': model,
            'wallclock_limit': wallclock_limit if epsilon_greedy else None
        }
        try:
            self.policy = policy(self.logger, **policy_kwargs)
            # noinspection PyUnresolvedReferences
            similarity_model = self.policy.mean
        except (KeyError, FileNotFoundError) as ex:
            self.logger.warning('Failed to initialize Policy: {}. Fallback to RandomSelection.'.format(ex))
            self.policy = RandomSelection(self.logger)
            similarity_model = None
        except AttributeError:
            # Raise if RandomPolicy is used
            similarity_model = None
        self.store = SimilarityStore(similarity_model)

    def fill_candidate(self, cs: CandidateStructure, ds: Dataset, worker: Worker = None,
                       cutoff: float = None, retries: int = 3) -> CandidateStructure:
        # Initialize tree if not exists
        with self.lock:
            if self.tree is None:
                self.tree = Tree(ds)
                self.store.add(ds.meta_features)

        if retries < 0:
            self.logger.warning('Retries exhausted. Aborting candidate sampling')
            return cs

        # traverse from root to a leaf node
        path, expand = self._select(force=retries == 0)

        result = None
        if expand or not path[-1].is_terminal():
            # expand last node in path if possible
            max_depths = 3
            max_failures = 3
            timeout = None if cutoff is None else timeit.default_timer() + cutoff
            for i in range(1, max_depths + 1):
                expansion, result, failures = self._expand(path, worker, cs.cid, timeout=timeout,
                                                           max_failures=max_failures,
                                                           include_preprocessing=i < max_depths)
                max_failures -= failures
                if expansion is not None:
                    path.append(expansion)
                    if expansion.is_terminal():
                        break
                else:
                    break
        else:
            self.logger.debug('Skipping MCTS expansion')

        if len(path) == 1:
            with self.lock:
                for node in path:
                    self.tree.get_node(node.id).exit()
            self.logger.warning(
                'Current path contains only ROOT node. Trying tree traversal {} more times...'.format(retries))
            return self.fill_candidate(cs, ds, worker=worker, retries=retries - 1)
        if not path[-1].is_terminal():
            with self.lock:
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
        if result is not None and result.structure_loss is not None and \
                result.structure_loss < util.worst_score(ds.metric)[-1]:
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

            with self.lock:
                node.enter()
                path.append(node)

                # Failsafe mechanism to enforce structure selection
                if force and node.is_terminal():
                    return path, False

                fully_expanded = self.tree.fully_expanded(node)
                if not fully_expanded:
                    return path, True

                # node is leaf
                if len(self.tree.get_children(node.id)) == 0:
                    return path, True

                candidate, candidate_score = self.policy.select(node, self.tree, force=force)  # descend a layer deeper
                # Current node is better than children or selected node failed
                if candidate.failed:
                    return path, not node.is_terminal()
                elif candidate_score <= score and node.is_terminal():
                    return path, False
                elif candidate_score <= score and not fully_expanded:
                    return path, True
                else:
                    node, score = candidate, candidate_score

    def _expand(self, nodes: List[Node],
                worker: Worker,
                cid: CandidateId,
                max_distance: float = 0.05,
                max_failures: int = 3,
                timeout: float = None,
                include_preprocessing: bool = True) -> Tuple[Optional[Node], Optional[Result], int]:
        node = nodes[-1]

        failure_count = 0
        n_actions = len(node.available_actions())
        while True:
            with self.lock:
                if node.is_terminal() and self.tree.fully_expanded(node):
                    return None, None, failure_count
                if timeout is not None and timeit.default_timer() > timeout:
                    self.logger.warning('Aborting expansion due to timeout')
                    return None, None, max_failures

                n_children = len(self.tree.get_children(node.id))
                if n_children >= n_actions:
                    break

                current_children = self.tree.get_children(node.id)
                action = self.policy.get_next_action(node, current_children,
                                                     include_preprocessing=include_preprocessing)
                if action is None:
                    return None, None, failure_count
                if failure_count >= max_failures:
                    self.logger.warning('Aborting expansion due to {} failed expansions'.format(failure_count))
                    return None, None, failure_count

                component = action()

                self.logger.debug(
                    '\tExpanding with {}. Option {}/{}'.format(component.name(), n_children + 1, n_actions))
                new_node = self.tree.add_node(estimator=action, parent_node=node)

            ds = node.ds
            config, key = self.cfg_cache.sample_configuration(
                configspace=component.get_hyperparameter_search_space(),
                mf=ds.meta_features,
                default=True)

            job = EvaluationJob(ds, cid.with_config(-new_node.id), cs=component, cutoff=self.cutoff, config=config,
                                cfg_keys=[key])
            result = worker.start_transform_dataset(job)

            if result.status.value == StatusType.SUCCESS.value:
                ds = Dataset(result.transformed_X, ds.y, ds.metric, ds.cutoff)
                new_node.partial_config = PartialConfig(key, config, str(new_node.id), ds.meta_features)
                new_node.ds = ds

                if ds.meta_features is None:
                    result.status = StatusType.CRASHED
                    result.structure_loss = util.worst_score(ds.metric)[-1]
                    new_node.failure_message = 'Missing MF'
                    # Currently only missing MF is counted as a failure
                    failure_count += 1
                else:
                    # Check if any node in the tree is similar to the new dataset
                    distance, idx = self.store.get_similar(ds.meta_features)
                    if np.allclose(node.ds.meta_features, ds.meta_features, equal_nan=True):
                        self.logger.debug('\t{} did not modify dataset'.format(component.name()))
                        result.status = StatusType.INEFFECTIVE
                        result.structure_loss = util.worst_score(ds.metric)[-1]
                        new_node.failure_message = 'Ineffective'
                    elif distance[0][0] <= max_distance:
                        # TODO: currently always the existing node is selected. This node could represent simpler model
                        self.logger.debug('\t{} produced a dataset similar to {}'.format(component.name(), idx[0][0]))
                        result.status = StatusType.DUPLICATE
                        result.structure_loss = util.worst_score(ds.metric)[-1]
                        new_node.failure_message = 'Duplicate {}'.format(idx[0][0])
                    else:
                        self.store.add(ds.meta_features)
                        node.enter()

                        # Hacky solution. If result loss is set, 'Incomplete' message is removed later
                        if result.structure_loss is None:
                            new_node.failure_message = None

                        if self.store_ds:
                            with open(os.path.join(self.workdir, '{}.pkl'.format(new_node.id)), 'wb') as f:
                                pickle.dump(ds, f)
            else:
                self.logger.debug(
                    '\t{} failed with default hyperparamter: {}'.format(component.name(), result.status))
                result.structure_loss = util.worst_score(ds.metric)[-1]
                if result.status == StatusType.TIMEOUT:
                    new_node.failure_message = 'Timeout'
                else:
                    new_node.failure_message = 'Crashed'

            if result.structure_loss is not None:
                n_children += 1
                self._backpropagate([key for key, values in new_node.steps], result.structure_loss)
                if result.structure_loss < util.worst_score(ds.metric)[-1]:
                    new_node.failure_message = None
                    result.partial_configs = [n.partial_config for n in nodes if n.partial_config is not None]
                    result.partial_configs.append(new_node.partial_config)
                    result.config = FlexiblePipeline(new_node.steps).configuration_space.get_default_configuration()

                    job.result = result
                    self.cfg_cache.register_result(job)
                    # Successful classifiers
                    return new_node, result, failure_count

            else:
                # Successful preprocessors
                return new_node, result, failure_count
        return None, None, failure_count

    def register_result(self, candidate: CandidateStructure, result: Result, update_model: bool = True,
                        **kwargs) -> None:
        reward = result.structure_loss

        if reward is None or not np.isfinite(reward):
            reward = 1

        try:
            self._backpropagate(candidate.pipeline.steps_.keys(), reward, exit=True)
        except (IndexError, ValueError, AttributeError) as ex:
            self.logger.warning('Unable to backpropagate results: {}'.format(ex))

    # noinspection PyMethodMayBeStatic
    def _backpropagate(self, path: List[str], reward: float, exit: bool = False) -> None:
        """Send the reward back up to the ancestors of the leaf"""
        # Also update root node

        ls = list(path)
        ls.append('0')
        with self.lock:
            for node_id in ls:
                node = self.tree.get_node(int(node_id))
                node.update(reward)
                if exit:
                    node.exit()

    def shutdown(self):
        if self.tree is None:
            self.logger.info("Search graph not initiated. Skipping rendering as pdf")
            return
        try:
            self.tree.plot(os.path.join(self.workdir, 'search_graph.pdf'))
        except ImportError as ex:
            self.logger.warning("Saving search graph is not possible. Please ensure that visualization "
                                "is correctly setup: {}".format(ex))
