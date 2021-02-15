import warnings

import pandas as pd
import numpy as np
import scipy
from scipy.stats import wilcoxon

warnings.filterwarnings("ignore", category=UserWarning)


def impute_missing(df: pd.DataFrame):
    def fill(row):
        if np.isnan(row['result']):
            if row['metric'] == 'auc':
                res = 0
            elif row['metric'] == 'logloss':
                res = 4
            else:
                raise ValueError('Unknown metric {}'.format(row['metric']))
        else:
            res = row['result']
        return res

    df['result'] = df.apply(fill, axis=1)


def compute_statistics(df: pd.DataFrame):
    count = df[(df['result'] == 0) | (df['result'] == 4)].groupby('task').agg({'id': 'count'})
    stats = df.groupby('task').agg({'result': [np.mean, np.std], 'metric': 'max'})
    result = pd.concat([stats, count], axis=1, join='outer').fillna(0)
    result.columns = ['mean', 'std', 'metric', 'missing']
    return result


def get_raw(idx: int):
    return raw[idx][raw[idx]['task'] == ds]['result']


tpot = pd.read_excel('results.xlsx', sheet_name=0)
autosklearn = pd.read_excel('results.xlsx', sheet_name=1)
dswizard = pd.read_excel('results.xlsx', sheet_name=2)
dswizard_star = pd.read_excel('results.xlsx', sheet_name=3)
rf = pd.read_excel('results.xlsx', sheet_name=4)

impute_missing(tpot)
impute_missing(autosklearn)
impute_missing(dswizard)
impute_missing(dswizard_star)
impute_missing(rf)

tpot2 = compute_statistics(tpot)
autosklearn2 = compute_statistics(autosklearn)
dswizard2 = compute_statistics(dswizard)
dswizard_star2 = compute_statistics(dswizard_star)
rf2 = compute_statistics(rf)

raw = [rf, autosklearn, tpot, dswizard, dswizard_star]
raw2 = [rf2, autosklearn2, tpot2, dswizard2, dswizard_star2]

tables = {}

for ds in dswizard2.index:
    if ds == 'collins':
        continue

    metric = tpot2.loc[ds]['metric']
    mean = np.array([df.loc[ds]['mean'] for df in raw2])
    std = np.array([df.loc[ds]['std'] for df in raw2])
    best, argbest = (np.max, np.argmax) if metric == 'auc' else (np.min, np.argmin)

    if metric not in tables:
        tables[metric] = [], []

    significance_ref = get_raw(argbest(mean))

    columns = ['{:15}'.format(str(ds).replace('_', '\\_')[:14])]
    rank = []
    for idx in range(len(mean)):
        rank.append(mean[idx])
        if mean[idx] in {0, 4}:
            columns.append('       ---                  ')
        else:
            entry = []
            if mean[idx] == best(mean):
                entry.append('\\B ')
                significant = True
            else:
                entry.append('   ')
                res = wilcoxon(significance_ref, get_raw(idx))
                significant = res.pvalue < 0.05
            if not significant:
                entry.append('\\ul{')
            else:
                entry.append('    ')
            entry.append('{:.3f} \\(\\pm\\) {:.3f}'.format(mean[idx], std[idx]))
            if not significant:
                entry.append('}')
            columns.append(''.join(entry))
    tables[metric][0].append('\t& '.join(columns) + '\t\\\\')
    tables[metric][1].append(rank)

for metric, data in tables.items():
    print('\n\n', metric, len(data[0]))
    print('\n'.join(data[0]))

    rank = np.array(data[1])
    if metric == 'auc':
        rank *= -1
    a = scipy.stats.rankdata(rank, axis=1)
    print('Avg. Rank\t& ' + '\t& '.join(['{:.3f}'.format(v) for v in np.mean(a, axis=0)]) + '\t\\\\')
