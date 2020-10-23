import warnings

import pandas as pd
import numpy as np
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

impute_missing(tpot)
impute_missing(autosklearn)

tpot2 = compute_statistics(tpot)
autosklearn2 = compute_statistics(autosklearn)

raw = [autosklearn, tpot]
raw2 = [autosklearn2, tpot2]

for ds in tpot2.index:
    metric = tpot2.loc[ds]['metric']
    mean = np.array([df.loc[ds]['mean'] for df in raw2])
    std = np.array([df.loc[ds]['std'] for df in raw2])
    best, argbest = (np.max, np.argmax) if metric == 'auc' else (np.min, np.argmin)

    significance_ref = get_raw(argbest(mean))

    print('{:40s}\t& '.format(str(ds) + ('*' if metric == 'auc' else '')), end='')

    for idx in range(len(mean)):
        if mean[idx] in {0, 4}:
            print('       ---                  \t& ', end='')
        else:
            if mean[idx] == best(mean):
                print('\\B ', end='')
                significant = False
            else:
                print('   ', end='')
                res = wilcoxon(significance_ref, get_raw(idx))
                significant = res.pvalue < 0.05
            if significant:
                print('\\ul{', end='')
            else:
                print('    ', end='')
            print('{:.4f} \\(\\pm\\) {:.4f}'.format(mean[idx], std[idx]), end='')
            if significant:
                print('}\t& ', end='')
            else:
                print(' \t& ', end='')
    print('\\\\')

a = 0
