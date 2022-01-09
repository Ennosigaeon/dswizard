import glob
import logging
import os
import timeit
from typing import List, Tuple, Optional

import joblib
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.utils import check_random_state

from dswizard.core.constants import MODEL_DIR
from dswizard.core.model import CandidateId, Dataset
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

        self._data: List[Tuple[float, CandidateId, FlexiblePipeline, np.ndarray]] = []
        self._n_classes = 0
        self.start = None

    def fit(self, ds: Dataset, fraction: float = 0.2):
        self.start = timeit.default_timer()
        self._n_classes = len(np.unique(ds.y))

        rs = StratifiedShuffleSplit(n_splits=1, test_size=fraction, random_state=0)
        train_idx, test_idx = next(rs.split(ds.X, ds.y))
        ds2 = Dataset(ds.X[test_idx], ds.y[test_idx], ds.metric)

        self._load(ds2)

        if self.n_bags > 0:
            self._build_bagged_ensemble(ds2)
        else:
            self._build_ensemble(ds2)

        self.logger.debug('Ensemble constructed')
        return self

    def _load(self, ds: Dataset):
        self.logger.debug('Loading models')
        models = {}

        for file in glob.glob(os.path.join(self.workdir, MODEL_DIR, '*.pkl')):
            with open(file, 'rb') as f:
                models[CandidateId.from_model_file(file)] = joblib.load(f)

        n_failed = 0
        for cid, model in models.items():
            try:
                y_prob = model.predict_proba(ds.X)
                y_pred = model.predict(ds.X)
                score = util.score(ds.y, y_prob, y_pred, ds.metric)
                self._data.append((score, cid, model, y_prob))
            except Exception:
                n_failed += 1
        self._data.sort(key=lambda x: x[0])
        self.logger.info(f'Loaded {len(self._data)} models. Failed to load {n_failed} models')

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

    def _ensemble_from_candidates(self, X, y, metric, candidates) -> Tuple[float, Optional[PrefitVotingClassifier]]:
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
            return np.inf, None

        scores = np.array([score for score, _ in cand_ensembles])
        idx = np.random.choice(np.where(scores == np.min(scores))[0])
        score, weights = cand_ensembles[idx]
        return score, PrefitVotingClassifier(
            [(clf[1].external_name, clf[2]) for i, clf in enumerate(candidates) if weights[i] > 0],
            weights=weights[weights > 0], voting='soft').fit(X, y)

    def _get_ensemble_score(self, y, metric, candidates, weights):
        n_models = weights.sum()
        y_probs = np.zeros((len(y), self._n_classes))

        for i in range(len(candidates)):
            y_probs += candidates[i][3] * weights[i]

        if n_models > 0:
            y_probs /= n_models
            score = util.score(y, y_probs, np.argmax(y_probs, axis=1), metric)
        else:
            score = util.worst_score(metric)[0]
        return score, y_probs

    @staticmethod
    def _score_with_model(y, metric, probs, n_models, candidate):
        n_models = float(n_models)
        new_probs = candidate[3]
        new_probs = (probs * n_models + new_probs) / (n_models + 1.0)
        y_pred = candidate[2]._final_estimator.classes_[np.argmax(new_probs, axis=1)]

        score = util.score(y, new_probs, y_pred, metric=metric)
        return score, new_probs

    def predict(self, X):
        return self.get_ensemble().predict(X)

    def predict_proba(self, X):
        return self.get_ensemble().predict_proba(X)

    def get_ensemble(self):
        return self.ensembles_[0][1]
