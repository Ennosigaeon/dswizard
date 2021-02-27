import glob
import itertools
import logging
import os
import timeit
from typing import List, Tuple

import joblib
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import check_random_state

from dswizard.core.model import Dataset
from dswizard.core.runhistory import RunHistory
from dswizard.pipeline.pipeline import FlexiblePipeline
from dswizard.pipeline.voting_ensemble import PrefitVotingClassifier
from dswizard.util import util


class EnsembleBuilder:

    def __init__(self,
                 workdir: str,
                 structure_fn: str,
                 cutoff: int = 900,
                 n_bags: int = 4,
                 bag_fraction: float = 0.75,
                 prune_fraction: float = 0.8,
                 min_models: int = 5,
                 max_models: int = 25,
                 random_state=None,
                 logger: logging.Logger = None):
        self.workdir = workdir
        self.structure_fn = structure_fn
        self.cutoff = cutoff

        self.n_bags = n_bags
        self.bag_fraction = bag_fraction
        self.prune_fraction = prune_fraction
        self.min_models = min_models
        self.max_models = max_models
        self.random_state = random_state

        if logger is None:
            self.logger = logging.getLogger('Ensemble')
        else:
            self.logger = logger

        self._data: List[Tuple[float, FlexiblePipeline, np.ndarray]] = []
        self._n_classes = 0
        self.start = None

    def fit(self, ds: Dataset, rh: RunHistory, fraction: float = 0.2):
        self.start = timeit.default_timer()
        self._n_classes = len(np.unique(ds.y))

        rs = StratifiedShuffleSplit(n_splits=1, test_size=fraction, random_state=0)
        train_idx, test_idx = next(rs.split(ds.X, ds.y))
        ds2 = Dataset(ds.X[test_idx], ds.y[test_idx], ds.metric)

        self._load(ds2, rh)

        if self.n_bags > 0:
            self._build_bagged_ensemble(ds2)
        else:
            self._build_ensemble(ds2)

        self.logger.debug('Ensemble constructed')
        return self

    def _load(self, ds: Dataset, rh: RunHistory):
        self.logger.debug('Loading models')
        steps = {}
        models = []

        # Load partial models with default hyperparameters
        for file in glob.glob(os.path.join(self.workdir, 'step_*.pkl')):
            with open(file, 'rb') as f:
                ls = joblib.load(f)
            steps[file.split('-')[1][:-4]] = ls

        runs = sorted(rh.get_all_runs(), key=lambda x: x[1].loss)
        n_models = min(self.max_models, max(int(len(runs) * (1.0 - self.prune_fraction)), self.min_models))
        runs = runs[:n_models]

        # Load models with tuned hyperparameters
        for cid, result in runs:
            file = os.path.join(self.workdir, 'models_{}-{}-{}.pkl'.format(*cid.as_tuple()))
            try:
                with open(file, 'rb') as f:
                    models += joblib.load(f)
            except FileNotFoundError:
                try:
                    partial = [steps[name] for name, _ in rh[cid.without_config()].steps]
                    for t in itertools.product(*partial):
                        pipeline = FlexiblePipeline(steps=[(str(idx), comp) for idx, comp in enumerate(t)])
                        pipeline.configuration = pipeline.get_hyperparameter_search_space(
                            None).get_default_configuration()
                        models.append(pipeline)
                except KeyError as ex:
                    self.logger.warning(
                        'File {} does not exist and pipeline step {} is not available'.format(file, str(ex)))

        n_failed = 0
        for model in models:
            try:
                y_prob = model.predict_proba(ds.X)
                y_pred = model.predict(ds.X)
                score = util.score(ds.y, y_prob, y_pred, ds.metric)
                self._data.append((score, model, y_prob))
            except Exception:
                n_failed += 1
        self._data.sort(key=lambda x: x[0])
        self.logger.info('Loaded {} models. Failed to load {} models'.format(len(self._data), n_failed))

    def _build_ensemble(self, ds: Dataset):
        self.logger.debug('Building ensemble')
        self.ensembles_ = [self._ensemble_from_candidates(ds.X, ds.y, ds.metric, self._data)]
        return self

    def _build_bagged_ensemble(self, ds: Dataset):
        self.logger.debug('Building bagged ensemble')

        n_models = len(self._data)
        bag_size = int(self.bag_fraction * n_models)

        self.ensembles_: List[Tuple[float, PrefitVotingClassifier]] = []
        rs = check_random_state(self.random_state)
        candidates = []
        for i in range(self.n_bags):
            cand_indices = rs.permutation(n_models)[:bag_size]
            candidates.append([self._data[ci] for ci in cand_indices])

        for c in candidates:
            score, ens = self._ensemble_from_candidates(ds.X, ds.y, ds.metric, c)
            if ens is not None:
                self.ensembles_.append((score, ens))
        self.ensembles_.sort(key=lambda x: x[0])
        return self

    def _ensemble_from_candidates(self, X, y, metric, candidates) -> Tuple[float, PrefitVotingClassifier]:
        weights = np.zeros(len(candidates))
        ens_score, ens_probs = self._get_ensemble_score(y, metric, candidates, weights)

        cand_ensembles = []
        for ens_count in range(self.max_models):
            new_scores = np.zeros(len(candidates))
            for idx, entry in enumerate(candidates):
                score, _ = self._score_with_model(y, metric, ens_probs, ens_count, entry)
                new_scores[idx] = score

                if timeit.default_timer() - self.start > self.cutoff:
                    self.logger.info('Aborting ensemble construction after timeout')
                    break
            else:
                idx = np.random.choice(np.where(new_scores == np.min(new_scores))[0])
                weights[idx] += 1
                ens_score, ens_probs = self._score_with_model(y, metric, ens_probs, ens_count, candidates[idx])

                cand_ensembles.append((ens_score, np.copy(weights)))
                continue
            break

        if len(cand_ensembles) == 0:
            return None, None

        scores = np.array([score for score, _ in cand_ensembles])
        idx = np.random.choice(np.where(scores == np.min(scores))[0])
        score, weights = cand_ensembles[idx]
        return score, PrefitVotingClassifier(
            [(str(i), clf[1]) for i, clf in enumerate(candidates) if weights[i] > 0],
            weights=weights[weights > 0], voting='soft').fit(X, y)

    def _get_ensemble_score(self, y, metric, candidates, weights):
        n_models = weights.sum()
        y_probs = np.zeros((len(y), self._n_classes))

        for i in range(len(candidates)):
            y_probs += candidates[i][2] * weights[i]

        if n_models > 0:
            y_probs /= n_models
            score = util.score(y, y_probs, np.argmax(y_probs, axis=1), metric)
        else:
            score = util.worst_score(metric)[0]
        return score, y_probs

    def _score_with_model(self, y, metric, probs, n_models, candidate):
        n_models = float(n_models)
        new_probs = candidate[2]
        new_probs = (probs * n_models + new_probs) / (n_models + 1.0)

        score = util.score(y, new_probs, np.argmax(new_probs, axis=1), metric=metric)
        return score, new_probs

    def predict(self, X):
        return self.get_ensemble().predict(X)

    def predict_proba(self, X):
        return self.get_ensemble().predict_proba(X)

    def get_ensemble(self):
        return self.ensembles_[0][1]
