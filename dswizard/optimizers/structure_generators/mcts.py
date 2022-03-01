import logging
import math
import operator
import os
import pickle
import threading
import timeit
from abc import ABC
from copy import deepcopy
from typing import List, Optional, Tuple, Type, Dict, Union, Any, Iterator

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
    EvaluationJob, Runtime
from dswizard.core.similaritystore import SimilarityStore
from dswizard.core.worker import Worker
from dswizard.pipeline.pipeline import FlexiblePipeline
from dswizard.util import util


class Node:
    """
    A representation of a single board state.
    MCTS works by constructing a tree of these Nodes.
    Could be e.g. a chess or checkers board state.
    """

    UNVISITED = 'Unvisited'
    INCOMPLETE = 'Incomplete'
    ROOT = 'Root'

    def __init__(self,
                 node_id: int,
                 ds: Optional[Dataset],
                 component: Optional[Type[EstimatorComponent]],
                 pipeline_prefix: List[Tuple[str, EstimatorComponent]] = None,
                 ):
        self.id = node_id
        self.ds = ds
        self.partial_config = None
        self.runtime: Optional[Runtime] = None

        self.failure_message: Optional[str] = None
        self.expanded = False

        if component is None:
            self.label = Node.ROOT
            self.component = None
        else:
            self.component = component()
            self.label = self.component.name(short=True)

        if pipeline_prefix is None:
            pipeline_prefix = []
        else:
            pipeline_prefix = deepcopy(pipeline_prefix)
            pipeline_prefix.append((f'{len(pipeline_prefix)}:{self.label}', self.component))
        self.steps: List[Tuple[str, EstimatorComponent]] = pipeline_prefix

        self.visits = 0
        self.reward = 0
        self.explanations: Dict[str, Dict[str, Any]] = {}

    def is_terminal(self):
        return self.component is not None and is_classifier(self.component)

    @property
    def failed(self):
        return self.failure_message is not None

    @property
    def unvisited(self):
        return self.failure_message is not None and self.failure_message == Node.UNVISITED

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
        if 'noop' in components:
            del components['noop']
        return components

    def enter(self, cid: CandidateId):
        self.reward += util.worst_score(self.ds.metric)[-1]
        self.explanations[cid.external_name]['selected'] = True

    def exit(self, cid: Optional[CandidateId] = None):
        self.reward -= util.worst_score(self.ds.metric)[-1]
        if cid:
            self.explanations[cid.external_name]['selected'] = False

    def update(self, reward: float) -> 'Node':
        self.visits += 1
        self.reward += reward
        return self

    def record_explanation(self, cid: CandidateId, score: float, policy: Dict):
        self.explanations[cid.external_name] = {
            'failure_message': self.failure_message,
            'score': score,
            'selected': False,
            'policy': policy
        }

    def __hash__(self) -> int:
        return self.id

    def __eq__(self, node2: int) -> bool:
        return self.id == node2


class Tree:
    ROOT: int = 0

    def __init__(self, ds: Dataset):
        self.G = nx.DiGraph()
        self.id_count = -1
        root = self.add_node(ds=ds)
        root.runtime = Runtime(0, 0)

        self.coef_progressive_widening = 0.7

    def add_node(self,
                 estimator: Optional[Type[EstimatorComponent]] = None,
                 ds: Optional[Dataset] = None,
                 parent_node: Node = None) -> Node:
        self.id_count += 1
        new_id = self.id_count
        node = Node(new_id, ds, estimator, pipeline_prefix=parent_node.steps if parent_node is not None else None)
        if ds is None:
            node.failure_message = Node.UNVISITED
        if parent_node is None:
            # ROOT node should always have at least 1 visit to be able to calculate UCT
            node.visits += 1

        self.G.add_node(new_id, value=node)
        if parent_node is not None:
            self.G.add_edge(parent_node, new_id)
        return node

    def inflate_node(self, estimator: Type[EstimatorComponent], parent_node: Node) -> Node:
        children = self.get_children(parent_node.id, include_unvisited=True)
        node = next(filter(lambda n: isinstance(n.component, estimator), children))
        assert node.failure_message == Node.UNVISITED
        node.failure_message = Node.INCOMPLETE
        return node

    def get_node(self, node: int) -> Node:
        return self.G.nodes[node]['value']

    def get_children(self, node: int, include_unvisited: bool = False) -> List[Node]:
        nodes = [self.G.nodes[n]['value'] for n in self.G.successors(node)]
        return list(filter(lambda n: include_unvisited or not n.unvisited, nodes))

    def expand_node(self, node: Node):
        for key, value in node.available_actions().items():
            self.add_node(value, parent_node=node)
        node.expanded = True

    def fully_expanded(self, node: Node, global_max: int = 7):
        # global_max is used as failsafe to prevent massive expansion of single node

        possible_children = len(node.available_actions())
        max_children = math.floor(math.pow(node.visits, self.coef_progressive_widening))
        max_children = min(max_children, possible_children, global_max)
        current_children = len(self.get_children(node.id))

        return possible_children == 0 or (current_children != 0 and current_children >= max_children)

    def predecessors(self, node: Node) -> Iterator:
        return self.G.predecessors(node.id)

    def plot(self, file: str):
        h = self.G.copy()
        unvisited_nodes = []
        for n, data in h.nodes(data=True):
            node: Node = data['value']

            score = node.reward / node.visits if node.visits > 0 else 0
            data['label'] = f'{node.label} ({n})\n{score:.4f} / {node.visits}'

            if node.failure_message == Node.UNVISITED:
                unvisited_nodes.append(n)
            elif node.failed:
                data['fillcolor'] = 'gray'
                data['style'] = 'filled'
                data['label'] += '\n' + node.failure_message

        h.remove_nodes_from(unvisited_nodes)
        nx.nx_agraph.to_agraph(h).draw(file, prog='dot')

    def __contains__(self, node: int) -> bool:
        return node in self.G.nodes


class Policy(ABC):

    def __init__(self, logger: logging.Logger, exploration_weight: float = 1, wallclock_limit: float = None, **kwargs):
        self.logger = logger
        self.exploration_weight = exploration_weight
        self._exploration_weight = exploration_weight
        self.start = timeit.default_timer()
        self.wallclock_limit = wallclock_limit

    def get_next_action(self,
                        node: Node,
                        children: List[Node],
                        cid: CandidateId,
                        include_preprocessing: bool = True,
                        include_classifier: bool = True) -> Optional[Type[EstimatorComponent]]:
        available_actions = node.available_actions()
        actions = []
        visited_children: List[Node] = list(filter(lambda _: not _.unvisited, children))

        # Prefer classifier if no children have been visited yet
        if include_classifier and len(visited_children) < 2:
            actions = self._get_actions(node, visited_children, include_preprocessing=False, include_classifier=True)
        if len(actions) == 0:
            actions = self._get_actions(node, visited_children, include_preprocessing=include_preprocessing,
                                        include_classifier=include_classifier)
        if len(actions) == 0:
            return None

        estimated_performances = dict(zip(actions, self.estimate_performance(actions, node.ds)))
        worst_performance = util.worst_score(node.ds.metric)[-1]
        for n in children:
            if not n.unvisited:
                continue

            name = n.component.component_name_
            performance = worst_performance
            if name in estimated_performances:
                performance = -estimated_performances[name]

            assert n.failure_message == Node.UNVISITED
            n.visits += 1
            n.reward += performance
            n.failure_message = None

            score, policy = self.uct(n, node, worst_score=worst_performance, decompose=True)
            estimated_performances[name] = score

            n.visits -= 1
            n.reward -= performance
            n.failure_message = Node.UNVISITED
            n.record_explanation(cid, score, policy)

        return available_actions[min(estimated_performances.items(), key=operator.itemgetter(1))[0]]

    def estimate_performance(self, actions: List[str], ds: Dataset, depth: int = 1) -> np.ndarray:
        pass

    def uct(self, n: Node, parent: Optional[Node], force: bool = False, worst_score: float = math.inf,
            decompose: bool = False) -> Union[float, Tuple[float, Dict[str, float]]]:
        """Upper confidence bound for trees"""

        # Always pick terminal node if forced
        if force and n.is_terminal():
            return -math.inf

        if parent is None or parent.visits == 0:
            log_N_vertex = 0
        else:
            log_N_vertex = math.log(parent.visits)

        if n.failed or n.visits == 0:
            exploitation = worst_score
            exploration = worst_score
        else:
            exploitation = n.reward / n.visits
            exploration = -math.sqrt(log_N_vertex / n.visits)
        score = exploitation + self._exploration_weight * exploration

        overfitting = 1 - (2 ** len(n.steps)) / (2 ** 4)
        adjusted_score = score * overfitting

        if decompose:
            return adjusted_score, {
                'exploit': exploitation,
                'explore': exploration,
                'visits': n.visits - 1,  # node visits is always off by 1 to prevent division by 0
                'overfit': overfitting,
                'weight': self._exploration_weight
            }
        else:
            return adjusted_score

    def select(self, node: Node, tree: Tree, force: bool = False) -> Tuple[Node, float]:
        """Select a child of node, balancing exploration & exploitation"""
        current_children = tree.get_children(node.id)
        if force:
            terminal_children = [child for child in current_children if child.is_terminal()]
            if len(terminal_children) > 0:
                children = terminal_children
            else:
                children = current_children
        else:
            children = current_children

        if self.wallclock_limit is not None:
            max_time = self.wallclock_limit
            passed_time = timeit.default_timer() - self.start
            self._exploration_weight = max(0.1, self.exploration_weight * (
                    (math.exp((max_time - passed_time) / max_time) - math.exp(0)) / (math.exp(1) - math.exp(0))
            ))

        worst_score = util.worst_score(node.ds.metric)[-1]
        scores = [self.uct(n, node, worst_score=worst_score) for n in children]
        idx = int(np.argmin(scores))

        # Selected non-existing child. Mark as "failed" to force abortion of select
        if idx >= len(current_children):
            children[idx].failure_message = 'Non Existing'

        return children[idx], scores[idx]

    @staticmethod
    def _get_actions(node: Node,
                     current_children: List[Node],
                     include_preprocessing: bool,
                     include_classifier: bool) -> List[str]:
        available_actions = node.available_actions(include_preprocessing=include_preprocessing,
                                                   include_classifier=include_classifier)
        exhausted_actions = [type(n.component) for n in current_children]
        actions = [key for key, value in available_actions.items() if value not in exhausted_actions]
        return actions


class RandomSelection(Policy):

    def estimate_performance(self, actions: List[str], ds: Dataset, depth: int = 1):
        return np.random.random(len(actions))


class TransferLearning(Policy):

    def __init__(self, logger: logging.Logger, model: str, **kwargs):
        super().__init__(logger, **kwargs)

        logger.info(f'Loading transfer model from {model}')
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

        # meta-learning base does not contain ordinal_encoder yet. Retrain meta-learning base with new components
        con = X[:, 42] == 'ordinal_encoder'
        X[con, 42] = 'one_hot_encoding'

        mean = self.mean.predict(X)
        var = self.var.predict(X)
        var = np.maximum(var, 0.01 * np.ones(var.shape))

        perf = np.random.multivariate_normal(mean, np.diag(var))
        return perf


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
        self.cid_to_node = {}
        self.wallclock_limit = wallclock_limit if epsilon_greedy else 1
        self.start = timeit.default_timer()

        if policy is None:
            if model is None:
                policy = RandomSelection
            else:
                policy = TransferLearning

        policy_kwargs = {
            'model': model,
            'wallclock_limit': wallclock_limit if epsilon_greedy else None
        }
        try:
            self.policy = policy(self.logger, **policy_kwargs)
            # noinspection PyUnresolvedReferences
            similarity_model = self.policy.mean
        except (KeyError, FileNotFoundError) as ex:
            self.logger.warning(f'Failed to initialize Policy: {ex}. Fallback to RandomSelection.')
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
                self.store.add(ds.meta_features, data=Tree.ROOT)

            self._record_explanations(cs.cid)

        if retries < 0:
            self.logger.warning('Retries exhausted. Aborting candidate sampling')
            return cs

        # traverse from root to a leaf node
        path, expand = self._select(cid=cs.cid, force=retries == 0)

        result = None
        if expand or not path[-1].is_terminal():
            # expand last node in path if possible
            max_depths = 3
            max_failures = 3
            timeout = None if cutoff is None or cutoff <= 0 else timeit.default_timer() + cutoff
            for i in range(1, max_depths + 1):
                expansion, result, failures = self._expand(path, worker, cs.cid,
                                                           timeout=timeout,
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

        failed = False
        if len(path) == 1:
            self.logger.warning(f'Current path contains only ROOT node. Trying tree traversal {retries} more times...')
            failed = True
        if not path[-1].is_terminal():
            self.logger.warning(f'Current path is not terminal. Trying tree traversal {retries} more times...')
            failed = True
        if failed:
            with self.lock:
                for node in path:
                    self.tree.get_node(node.id).exit(cs.cid)
            return self.fill_candidate(cs, ds, worker=worker, retries=retries - 1)

        # A simulation is not necessary. Simulated results are already incorporated in the policy

        node = path[-1]
        self.cid_to_node[cs.cid] = node
        self.logger.debug(f'Sampled pipeline structure: {node.steps}')
        pipeline = FlexiblePipeline(node.steps)

        cs.configspace = pipeline.configuration_space
        cs.pipeline = pipeline
        cs.cfg_keys = [n.partial_config.cfg_key for n in path[1:]]

        # If no simulation was necessary, add the default configuration as first result
        if result is not None and result.structure_loss is not None and \
                result.structure_loss < util.worst_score(ds.metric)[-1]:
            cs.add_result(result)

        assert len(cs.steps) == len(cs.cfg_keys)
        return cs

    def _select(self, cid: CandidateId, force: bool = False) -> Tuple[List[Node], bool]:
        """Find an unexplored descendent of ROOT"""

        path: List[Node] = []
        node = self.tree.get_node(Tree.ROOT)
        score = -math.inf
        epsilon = (timeit.default_timer() - self.start) / self.wallclock_limit

        # Failsafe mechanism to enforce structure selection
        if force or np.random.random() < epsilon / 2:
            nodes = [self.tree.get_node(n) for n in nx.dfs_tree(self.tree.G, self.tree.ROOT)]
            terminal_nodes = [n for n in nodes if n.is_terminal() and not n.failed]
            probs = np.array([n.reward / n.visits for n in terminal_nodes]) * -1
            node = np.random.choice(terminal_nodes, 1, p=probs / np.sum(probs))[0]

            path = nx.shortest_path(self.tree.G, source=self.tree.ROOT, target=node.id)
            # noinspection PyTypeChecker
            path[0] = self.tree.get_node(path[0])
            # noinspection PyTypeChecker
            path[-1] = self.tree.get_node(path[-1])

            with self.lock:
                for node in path:
                    node.enter(cid)
            return path, False

        while True:
            self.logger.debug(f'\tSelecting {node.label}')
            if node.failed:
                self.logger.warning(
                    'Selected node has no dataset, meta-features or partial configurations. This should not happen')

            with self.lock:
                node.enter(cid)
                path.append(node)

                # We never expand if force is given
                fully_expanded = self.tree.fully_expanded(node)
                if not fully_expanded:
                    return path, True

                # node is leaf
                if len(self.tree.get_children(node.id)) == 0:
                    return path, node.is_terminal()

                candidate, candidate_score = self.policy.select(node, self.tree)
                # Current node is better than children or selected node failed
                if candidate.failed:
                    return path, not node.is_terminal()
                elif candidate_score >= score and node.is_terminal():
                    return path, False
                elif candidate_score >= score and not fully_expanded:
                    return path, not node.is_terminal()
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

                if not node.expanded:
                    self.tree.expand_node(node)
                n_children = len(self.tree.get_children(node.id))
                if n_children >= n_actions:
                    break

                children = self.tree.get_children(node.id, include_unvisited=True)
                action = self.policy.get_next_action(node, children, cid, include_preprocessing=include_preprocessing)
                if action is None:
                    return None, None, failure_count
                if failure_count >= max_failures:
                    self.logger.warning(f'Aborting expansion due to {failure_count} failed expansions')
                    return None, None, failure_count

                new_node = self.tree.inflate_node(estimator=action, parent_node=node)
                component = new_node.component
                self.logger.debug(f'\tExpanding with {component.name()}. Option {n_children + 1}/{n_actions}')

            ds = node.ds
            config, key = self.cfg_cache.sample_configuration(
                cid=cid.with_config(0),
                name=new_node.steps[-1][0],
                configspace=component.get_hyperparameter_search_space(),
                mf=ds.meta_features,
                default=True)

            job = EvaluationJob(ds, cid.with_config(f'{len(node.steps)}_{component.name(short=True)}'),
                                cs=component, cutoff=self.cutoff, config=config, cfg_keys=[key])
            result = worker.start_transform_dataset(job)

            if result.status.value == StatusType.SUCCESS.value:
                ds = Dataset(result.transformed_X, ds.y, ds.metric, ds.cutoff)
                new_node.partial_config = PartialConfig(key, config, str(new_node.id), ds.meta_features)
                new_node.ds = ds

                # Add trainings time of all previous nodes to new node
                result.runtime.training_time += sum([n.runtime.training_time for n in nodes])
                new_node.runtime = result.runtime

                if ds.meta_features is None:
                    result.status = StatusType.CRASHED
                    result.structure_loss = util.worst_score(ds.metric)[-1]
                    new_node.failure_message = 'Missing MF'
                    # Currently only missing MF is counted as a failure
                    failure_count += 1
                else:
                    # Check if any node in the tree is similar to the new dataset
                    distance, _, idx = self.store.get_similar(ds.meta_features)
                    if np.allclose(node.ds.meta_features, ds.meta_features, equal_nan=True):
                        self.logger.debug(f'\t{component.name()} did not modify dataset')
                        result.status = StatusType.INEFFECTIVE
                        result.structure_loss = util.worst_score(ds.metric)[-1]
                        new_node.failure_message = 'Ineffective'
                    elif distance <= max_distance:
                        # TODO: currently always the existing node is selected. This node could represent simpler model
                        self.logger.debug(f'\t{component.name()} produced a dataset similar to {idx}')
                        result.status = StatusType.DUPLICATE
                        result.structure_loss = util.worst_score(ds.metric)[-1]
                        new_node.failure_message = f'Duplicate {idx}'
                    else:
                        self.store.add(ds.meta_features, data=new_node.id)
                        # Enter node as enter was not called during tree traversal yet
                        new_node.enter(cid)
                        new_node.failure_message = None

                        if self.store_ds:
                            with open(os.path.join(self.workdir, f'{new_node.id}.pkl'), 'wb') as f:
                                pickle.dump(ds, f)
            else:
                self.logger.debug(f'\t{component.name()} failed with default hyperparamter: {result.status}')
                result.structure_loss = util.worst_score(ds.metric)[-1]
                if result.status == StatusType.TIMEOUT:
                    new_node.failure_message = 'Timeout'
                else:
                    new_node.failure_message = 'Crashed'

            if result.structure_loss is not None:
                n_children += 1
                self._backpropagate(new_node, result.structure_loss)
                if result.structure_loss < util.worst_score(ds.metric)[-1]:
                    result.partial_configs = [n.partial_config for n in nodes[1:]]
                    result.partial_configs.append(new_node.partial_config)
                    config = FlexiblePipeline(new_node.steps).configuration_space.get_default_configuration()
                    config.origin = 'Default'
                    result.config = config

                    job.result = result
                    self.cfg_cache.register_result(job)
                    # Successful classifiers
                    return new_node, result, failure_count

            else:
                # Successful preprocessors
                return new_node, result, failure_count
        return None, None, failure_count

    def _record_explanations(self, cid: CandidateId):
        root = self.tree.get_node(self.tree.ROOT)
        worst_score = util.worst_score(root.ds.metric)[-1]
        for node_id in self.tree.G.nodes:
            node = self.tree.get_node(node_id)
            if node.visits == 0:
                # Exclude unvisited nodes from tree traversal
                continue
            try:
                parent = next(self.tree.G.predecessors(node_id))
            except StopIteration:
                parent = None
            score, policy = self.policy.uct(node, parent, worst_score=worst_score, decompose=True)
            node.record_explanation(cid, score, policy)

    def register_result(self, candidate: CandidateStructure, result: Result, update_model: bool = True,
                        **kwargs) -> None:
        reward = result.structure_loss

        if reward is None or not np.isfinite(reward):
            reward = util.worst_score(self.tree.get_node(self.tree.ROOT).ds.metric)[-1]

        try:
            node = self.cid_to_node[candidate.cid]
            self._backpropagate(node, reward, exit_=True)
        except (IndexError, ValueError, AttributeError, KeyError) as ex:
            self.logger.warning(f'Unable to backpropagate results: {ex}')

    # noinspection PyMethodMayBeStatic
    def _backpropagate(self, node: Node, reward: float, exit_: bool = False) -> None:
        """Send the reward back up to the ancestors of the leaf"""
        with self.lock:
            nodes = [node] + list(self.tree.predecessors(node)) + [self.tree.get_node(self.tree.ROOT)]

            for pred in nodes:
                pred.update(reward)
                if exit_:
                    pred.exit()

    def explain(self) -> Dict[str, Any]:
        with self.lock:
            nodes = {}
            edges = nx.dfs_successors(self.tree.G, source=Tree.ROOT)

            for node_id in self.tree.G.nodes:
                node = self.tree.get_node(node_id)
                nodes[node_id] = {
                    'label': node.label,
                    'details': node.explanations
                }

            def transform_node(node_id: int):
                children = []
                element = {
                    'id': str(node_id),
                    **nodes[node_id]
                }
                try:
                    for child in edges[node_id]:
                        children.append(transform_node(child))
                except KeyError:
                    pass
                if len(children) > 0:
                    element['children'] = children
                return element

            hierarchy = transform_node(Tree.ROOT)
            return hierarchy

    def shutdown(self):
        if self.tree is None:
            self.logger.info("Search graph not initiated. Skipping rendering as pdf")
            return
        try:
            self.tree.plot(os.path.join(self.workdir, 'search_graph.pdf'))
        except ImportError as ex:
            self.logger.warning("Saving search graph is not possible. Please ensure that visualization "
                                f"is correctly setup: {ex}")
