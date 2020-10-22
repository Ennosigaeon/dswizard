import glob
import itertools
import json
import logging
import os
from typing import List, Tuple

import joblib
import numpy as np
from joblib import Parallel, delayed
from sklearn.utils import check_random_state

from dswizard.components.pipeline import FlexiblePipeline
from dswizard.components.voting_ensemble import PrefitVotingClassifier
from dswizard.core.model import Dataset, CandidateStructure
from dswizard.util import util


class EnsembleBuilder:

    def __init__(self,
                 workdir: str,
                 structure_fn: str,
                 n_bags=20,
                 bag_fraction=0.25,
                 prune_fraction=0.8,
                 max_models=50,
                 random_state=None,
                 logger: logging.Logger = None):
        self.workdir = workdir
        self.structure_fn = structure_fn

        self.n_bags = n_bags
        self.bag_fraction = bag_fraction
        self.prune_fraction = prune_fraction
        self.max_models = max_models
        self.random_state = random_state

        if logger is None:
            self.logger = logging.getLogger('Ensemble')
        else:
            self.logger = logger

        self._data: List[Tuple[float, FlexiblePipeline, np.ndarray]] = []
        self._n_classes = 0

    def fit(self, ds: Dataset):
        """Perform model fitting and ensemble building"""
        self._load(ds)
        self._build_ensemble(ds)
        return self

    def _load(self, ds: Dataset):
        self.logger.debug('Loading models')
        steps = {}
        models = []
        # Load models with tuned hyperparameters
        for file in glob.glob(os.path.join(self.workdir, 'models_*.pkl')):
            with open(file, 'rb') as f:
                models += joblib.load(f)

        # Load partial models with default hyperparameters
        for file in glob.glob(os.path.join(self.workdir, 'step_*.pkl')):
            with open(file, 'rb') as f:
                ls = joblib.load(f)
            steps[file.split('-')[1][:-4]] = ls

        # Merge with actual structures to obtain fitted pipelines with default hyperparameters
        with open(self.structure_fn, 'r') as structure_file:
            for line in structure_file:
                raw = json.loads(line)
                cs = CandidateStructure.from_dict(raw)
                partial = [steps[name] for name, _ in cs.steps]
                for t in itertools.product(*partial):
                    models.append(FlexiblePipeline(steps=[(str(idx), comp) for idx, comp in enumerate(t)]))

        for model in models:
            y_prob = model.predict_proba(ds.X)
            y_pred = model.predict(ds.X)
            score = util.score(ds.y, y_prob, y_pred, ds.metric)
            self._data.append((score, model, y_prob))
        self._data.sort(key=lambda x: x[0])
        self.logger.info('Loaded {} models'.format(len(self._data)))

    def _build_ensemble(self, ds: Dataset):
        self.logger.debug('Building ensemble')
        self._n_classes = len(np.unique(ds.y))

        n_models = min(self.max_models, int(len(self._data) * (1.0 - self.prune_fraction)))
        bag_size = int(self.bag_fraction * n_models)

        self.ensembles_: List[Tuple[float, PrefitVotingClassifier]] = []
        rs = check_random_state(self.random_state)
        candidates = []
        for i in range(self.n_bags):
            cand_indices = rs.permutation(n_models)[:bag_size]
            candidates.append(np.array([self._data[ci] for ci in cand_indices]))

        self.ensembles_ = Parallel()(
            delayed(self._ensemble_from_candidates)(ds.X, ds.y, ds.metric, c)
            for c in candidates
        )
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

            idx = np.random.choice(np.where(new_scores == np.min(new_scores))[0])
            weights[idx] += 1
            ens_score, ens_probs = self._score_with_model(y, metric, ens_probs, ens_count, candidates[idx, :])

            cand_ensembles.append((ens_score, np.copy(weights)))

        scores = np.array([score for score, _ in cand_ensembles])
        idx = np.random.choice(np.where(scores == np.min(scores))[0])
        score, weights = cand_ensembles[idx]
        return score, PrefitVotingClassifier(
            [(str(i), clf) for i, clf in enumerate(candidates[weights > 0][:, 1].tolist())],
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
