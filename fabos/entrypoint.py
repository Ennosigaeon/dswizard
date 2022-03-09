import argparse
import logging
import pickle
import textwrap

import pandas as pd

from dswizard.core.master import Master
from dswizard.core.model import Dataset
from dswizard.util import util

parser = argparse.ArgumentParser(epilog=textwrap.dedent('''
dswizard

Fits an ML pipeline to the provided data set. Start via
`docker run --rm -it -v <LOCAL_DATA_DIR>:/dswizard dswizard --data <DATA_FILE>`
'''))
parser.add_argument('--wallclock_limit', type=float, help='Maximum optimization time for in seconds', default=600)
parser.add_argument('--cutoff', type=float, help='Maximum cutoff time for a single evaluation in seconds', default=60)
parser.add_argument('--data', type=str, help='File containing data set', required=True)
args = parser.parse_args()

util.setup_logging('/dswizard/log.txt')
logger = logging.getLogger()

X = pd.read_csv(args.data)
y = X[X.columns[-1]]
X.drop(columns=X.columns[-1], inplace=True)

ds = Dataset(X.values, y.values, cutoff=args.cutoff, feature_names=X.columns.tolist())

master = Master(
    ds=ds,
    working_directory='/dswizard/workdir',
    model='/opt/dswizard/rf_complete.pkl',
    wallclock_limit=args.wallclock_limit,
    cutoff=args.cutoff,
)

pipeline, run_history, ensemble = master.optimize()
_, incumbent = run_history.get_incumbent()

logging.info(f'Best found configuration: {incumbent.steps}\n'
             f'{incumbent.get_incumbent().config} with loss {incumbent.get_incumbent().loss}')
logging.info(f'A total of {len(run_history.data)} unique structures where sampled.')
logging.info(f'A total of {len(run_history.get_all_runs())} runs where executed.')

y_pred = pipeline.predict(ds.X)
y_prob = pipeline.predict_proba(ds.X)

logging.info(f'Final pipeline:\n{pipeline}')
logging.info(f'Final test performance {util.score(ds.y, y_prob, y_pred, ds.metric)}')
logging.info(f'Final ensemble performance '
             f'{util.score(ds.y, ensemble.predict_proba(ds.X), ensemble.predict(ds.X), ds.metric)} '
             f'based on {len(ensemble.estimators_)} individuals')

logging.info('Storing results in \'/dswizard/results.pkl\'')
with open('/dswizard/results.pkl', 'wb') as f:
    pickle.dump((pipeline, run_history, ensemble), f)
