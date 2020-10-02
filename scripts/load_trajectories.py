import pickle
import numpy as np
from scipy.stats import rankdata

from smac.runhistory.runhistory import RunHistory

import matplotlib.pyplot as plt


def load_autosklearn():
    with open('autosklearn/configspace.pkl', 'rb') as f:
        cs = pickle.load(f)

    with open('autosklearn/start.txt', 'r') as f:
        start = float(f.readline())

    rh = RunHistory(lambda x: x)
    rh.load_json('autosklearn/runhistory_0.json', cs)

    scores = []
    for value in rh.data.values():
        scores.append((value.additional_info + value.time - start, value.cost))
    scores.insert(0, (0, 0))
    scores = np.array(scores)
    scores[:, 1] = np.maximum.accumulate(scores[:, 1])
    return scores


def load_tpot():
    with open('tpot/runhistory.pkl', 'rb') as f:
        rh = pickle.load(f)

    scores = []
    for time, score, pipeline in rh:
        scores.append((time, score))
    scores.insert(0, (0, 0))
    scores = np.array(scores)
    scores[:, 1] = np.maximum.accumulate(scores[:, 1])
    return scores


def compute_rank(data):
    x_ticks = np.arange(0, 3660, 1)

    rasterized = np.zeros((len(x_ticks), len(data)))
    idx = np.zeros(len(data), dtype=int)
    for i, x in enumerate(x_ticks):
        for j, d in enumerate(data):
            if idx[j] + 1 < len(data[j]) and data[j][idx[j] + 1][0] < x:
                idx[j] += 1
            rasterized[i][j] = data[j][idx[j]][1]

    rank = rankdata(rasterized, axis=1)
    return rank


auto_sklearn = load_autosklearn()
tpot = load_tpot()

x_max = max(auto_sklearn[:, 0].max(), tpot[:, 0].max())
auto_sklearn = np.vstack((auto_sklearn, np.array([[x_max, auto_sklearn[-1, 1]]])))
tpot = np.vstack((tpot, np.array([[x_max, tpot[-1, 1]]])))

rank = compute_rank([auto_sklearn, tpot])

fig = plt.figure()
ax = fig.add_subplot(111)

ax.step(auto_sklearn[:, 0], auto_sklearn[:, 1], label='auto-sklearn')
ax.step(tpot[:, 0], tpot[:, 1], label='tpot')
ax.legend()
fig.savefig('fig/trajectory.png')
