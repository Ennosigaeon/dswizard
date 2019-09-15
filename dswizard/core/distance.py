import abc
from typing import Tuple, List

import numpy as np
import scipy.integrate
from sklearn.neighbors import KernelDensity


class KdeDistribution:

    def __init__(self, samples: np.ndarray, bandwidth: float = 1):
        if samples.ndim != 1:
            raise ValueError('Dimensionality has to be exactly 1')

        self.samples = np.sort(samples)
        self.min_value = self.samples[0]
        self.max_value = self.samples[-1]

        # TODO KernelDensity stores copy of dataset. Probably very high memory consumption
        self.kde = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(samples.reshape(-1, 1))

    def evaluate(self, x: float):
        y_log = self.kde.score_samples(np.array([[x]]))
        return np.exp(y_log)


class Distance(abc.ABC):

    @staticmethod
    def compute(a: np.ndarray, b: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Computes the minimum average Squared-Integrated-Distance between two 2d vectors.

        1. Fill a matrix using a distance metric for 1d-vectors
        2. Use dynamic programming to compute minimum cost path
        3. Return average distance over all features
        """

        np.atleast_2d(a)
        np.atleast_2d(b)

        dist_a = [KdeDistribution(a[:, i]) for i in range(a.shape[1])]
        dist_b = [KdeDistribution(b[:, i]) for i in range(b.shape[1])]
        return Distance.compute_dist(dist_a, dist_b)

    @staticmethod
    def compute_dist(dist_a: List[KdeDistribution], dist_b: List[KdeDistribution]) -> Tuple[float, np.ndarray]:
        n_a = len(dist_a)
        n_b = len(dist_b)

        M = np.zeros((n_a, n_b))

        for i in range(n_a):
            for j in range(n_b):
                M[i][j] = SquaredIntegratedDistance.pair_wise(dist_a[i], dist_b[j])

        cost, phi = SquaredIntegratedDistance.min_cost(M)
        return cost / (n_a + n_b), phi

    @staticmethod
    def min_cost(M: np.ndarray, pen: float = 1):
        # https://github.com/dpwe/dp_python

        D = np.zeros(M.shape, dtype=np.float)
        phi = np.zeros(M.shape, dtype=np.int)

        # initialize bottom left
        D[0, 0] = M[0, 0]
        phi[0, 0] = 0

        # bottom edge can only come from preceding column
        D[0, 1:] = M[0, 0] + np.cumsum(M[0, 1:] + pen)
        phi[0, 1:] = 1

        # left edge can only come from preceding row
        D[1:, 0] = M[0, 0] + np.cumsum(M[1:, 0] + pen)
        phi[1:, 0] = 2

        for c in range(1, np.shape(M)[1]):
            for r in range(1, np.shape(M)[0]):
                best_preceding_costs = [D[r - 1, c - 1], pen + D[r, c - 1], pen + D[r - 1, c]]
                # noinspection PyTypeChecker
                tb: int = np.argmin(best_preceding_costs)
                D[r, c] = best_preceding_costs[tb] + M[r, c]
                phi[r, c] = tb

        return D[-1, -1], phi

    @staticmethod
    def traceback(phi: np.ndarray):
        # https://github.com/dpwe/dp_python

        i = phi.shape[0] - 1
        j = phi.shape[1] - 1

        idx = [(i, j)]
        # Work backwards until we get to starting point (0, 0)
        while i >= 0 and j >= 0:
            tb = phi[i, j]
            if tb == 0:
                i -= 1
                j -= 1
            elif tb == 1:
                j -= 1
            elif tb == 2:
                i -= 1
            idx.insert(0, (i, j))
        return idx[1:]

    @staticmethod
    @abc.abstractmethod
    def pair_wise(dist_a: KdeDistribution, dist_b: KdeDistribution) -> float:
        pass


class SquaredIntegratedDistance(Distance):

    @staticmethod
    def pair_wise(dist_a: KdeDistribution, dist_b: KdeDistribution) -> float:
        """
        Computes the Squared-Integrated-Distance between two 1d vectors.
        """
        lower = min(dist_a.min_value, dist_b.min_value)
        upper = max(dist_a.max_value, dist_b.max_value)

        y = scipy.integrate.quad(lambda x: (dist_a.evaluate(x) - dist_b.evaluate(x)) ** 2, lower, upper)[0]
        norm1 = scipy.integrate.quad(lambda x: dist_a.evaluate(x) ** 2, lower, upper)[0]
        norm2 = scipy.integrate.quad(lambda x: dist_b.evaluate(x) ** 2, lower, upper)[0]

        return y / (norm1 + norm2)
