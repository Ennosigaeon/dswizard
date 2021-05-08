import os
import pickle
from argparse import ArgumentParser

import numpy as np
from scipy.stats import rankdata

from smac.runhistory.runhistory import RunHistory

import matplotlib.pyplot as plt

min_time = 0
max_time = 3660

worst_score = (0, 1)
factor = 1


def _format(scores):
    scores.insert(0, (min_time, worst_score[0]))
    scores = np.array(scores)
    scores[:, 1] = np.clip(scores[:, 1], min(*worst_score), max(*worst_score))
    scores[:, 1] = factor * scores[:, 1]
    # scores[:, 1] = np.maximum.accumulate(scores[:, 1])
    return scores


def load_autosklearn(base_dir: str):
    try:
        with open(os.path.join(base_dir, 'configspace.pkl'), 'rb') as f:
            cs = pickle.load(f)

        with open(os.path.join(base_dir, 'start.txt'), 'r') as f:
            start = float(f.readline())

        rh = RunHistory(lambda x: x)
        rh.load_json(os.path.join(base_dir, 'runhistory_0.json'), cs)
        scores = []
        for value in rh.data.values():
            scores.append((value.additional_info + value.time - start, value.cost))
    except FileNotFoundError:
        scores = [(min_time, worst_score[0]), (max_time, worst_score[0])]

    return _format(scores)


def load_tpot(base_dir):
    try:
        with open(os.path.join(base_dir, 'runhistory.pkl'), 'rb') as f:
            rh = pickle.load(f)
            scores = []
            for time, score, pipeline in rh:
                scores.append((time, score))
    except FileNotFoundError:
        scores = [(min_time, worst_score[0]), (max_time, worst_score[0])]
    return _format(sorted(scores))


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


parser = ArgumentParser()
parser.add_argument('base_dir', type=str, help='Base dir containing raw results')
args = parser.parse_args()
base_dir = args.base_dir

for task, metric in [('ada_agnostic', 'auc'),
                     ('adult', 'auc'),
                     ('analcatdata_authorship', 'logloss'),
                     ('analcatdata_dmft', 'logloss'),
                     ('Australian', 'auc'),
                     ('eeg-eye-state', 'auc'),
                     ('mfeat-morphological', 'logloss'),
                     ('segment', 'logloss'),
                     ('sylvine', 'auc'),
                     ('vehicle', 'logloss')]:
    if metric == 'logloss':
        factor = -1
        worst_score = (5, 0)
    else:
        factor = 1
        worst_score = (0, 1)

    for fold in range(10):
        auto_sklearn = load_autosklearn(os.path.join(base_dir, 'autosklearn', task, str(fold)))
        tpot = load_tpot(os.path.join(base_dir, 'tpot', task, str(fold)))

        rank = compute_rank([auto_sklearn, tpot])

        fig = plt.figure()
        ax = fig.add_subplot(111)
        ax.scatter(auto_sklearn[:, 0], auto_sklearn[:, 1], label='auto-sklearn')
        ax.step(auto_sklearn[:, 0], np.maximum.accumulate(auto_sklearn[:, 1]), label='auto-sklearn', where='post')
        ax.scatter(tpot[:, 0], tpot[:, 1], label='tpot')
        ax.step(tpot[:, 0], np.maximum.accumulate(tpot[:, 1]), label='tpot', where='post')
        ax.legend()
        fig.savefig(f'fig/{task}_{fold}.png')
        plt.close(fig)
