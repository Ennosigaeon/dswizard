from typing import Optional

import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline

from dswizard.components.meta_features import MetaFeatures


class SimilarityStore:
    N_MF = 44

    def __init__(self, model: Optional[Pipeline]):
        self.mfs = None
        if model is not None:
            self.model: Pipeline = model
            # Remove OHE encoded algorithm
            self.weight = self.model.steps[-1][1].feature_importances_[0:SimilarityStore.N_MF]
            self.weight = (self.weight / self.weight.sum()) * SimilarityStore.N_MF
        else:
            self.model = None
            self.weight = np.ones(SimilarityStore.N_MF)
        self.neighbours = NearestNeighbors(metric='wminkowski', p=2, metric_params={'w': self.weight})

    def add(self, mf: MetaFeatures):
        mf_normal = self._normalize(mf)
        if self.mfs is None:
            self.mfs = mf_normal.reshape(1, -1)
        else:
            self.mfs = np.append(self.mfs, mf_normal, axis=0)
        self.neighbours.fit(self.mfs)

    def get_similar(self, mf: MetaFeatures):
        X = self._normalize(mf)

        return self.neighbours.kneighbors(X, n_neighbors=1)

    def _normalize(self, X):
        # remove unused MF
        Xt = X[:, 0:SimilarityStore.N_MF]
        if self.model is not None:
            for name, transform in self.model.steps[:2]:
                Xt = transform.transform(Xt)
        return Xt
