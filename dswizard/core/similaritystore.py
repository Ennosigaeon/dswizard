import numpy as np
from sklearn.neighbors import NearestNeighbors

from automl.components.meta_features import MetaFeatures


# TODO replace KNN with something that uses feature importance
class SimilarityStore:

    def __init__(self):
        self.neighbours = NearestNeighbors()
        self.mfs = None

    def add(self, mf: MetaFeatures):
        if self.mfs is None:
            self.mfs = mf
        else:
            self.mfs = np.append(self.mfs, mf, axis=0)
        self.neighbours.fit(self.mfs)

    def get_similar(self, mf: MetaFeatures):
        return self.neighbours.kneighbors(mf, n_neighbors=1)
