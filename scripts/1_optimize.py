"""
Example 1 - Single Threaded
================================

"""
import argparse
import logging
import os

import openml

from dswizard.core.master import Master
from dswizard.core.model import Dataset
from dswizard.optimizers.bandit_learners.pseudo import PseudoBandit
from dswizard.optimizers.config_generators import Hyperopt
from dswizard.optimizers.structure_generators.mcts import MCTS, TransferLearning
from dswizard.util import util

parser = argparse.ArgumentParser(description='Example 1 - dswizard optimization.')
parser.add_argument('--wallclock_limit', type=float, help='Maximum optimization time for in seconds', default=300)
parser.add_argument('--cutoff', type=float, help='Maximum cutoff time for a single evaluation in seconds', default=60)
parser.add_argument('--log_dir', type=str, help='Directory used for logging', default='run/')
parser.add_argument('--fold', type=int, help='Fold of OpenML task to optimize', default=0)
parser.add_argument('task', type=int, help='OpenML task id')
args = parser.parse_args()

util.setup_logging(os.path.join(args.log_dir, str(args.task), 'log.txt'))
logger = logging.getLogger()
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# Load dataset
# Tasks: 18, 53, 9983, 146822, 168912
logger.info('Processing task {}'.format(args.task))
task = openml.tasks.get_task(args.task)
train_indices, test_indices = task.get_train_test_split_indices(repeat=0, fold=args.fold, sample=0)

# noinspection PyUnresolvedReferences
X, y, _, _ = task.get_dataset().get_data(task.target_name)
X_train = X.loc[train_indices]
y_train = y[train_indices]
X_test = X.loc[test_indices]
y_test = y[test_indices]

ds = Dataset(X_train.to_numpy(), y_train.to_numpy(), metric='rocauc')
ds_test = Dataset(X_test.to_numpy(), y_test.to_numpy(), metric=ds.metric)

master = Master(
    ds=ds,
    working_directory=os.path.join(args.log_dir, str(args.task)),
    n_workers=2,
    model='../dswizard/assets/rf_complete.pkl',

    wallclock_limit=args.wallclock_limit,
    cutoff=args.cutoff,
    pre_sample=False,

    config_generator_class=Hyperopt,

    structure_generator_class=MCTS,
    structure_generator_kwargs={'policy': TransferLearning},

    bandit_learner_class=PseudoBandit
)

pipeline, run_history, ensemble = master.optimize()

# Analysis
_, incumbent = run_history.get_incumbent()

logging.info('Best found configuration: {}\n{} with loss {}'.format(incumbent.steps,
                                                                    incumbent.get_incumbent().config,
                                                                    incumbent.get_incumbent().loss))
logging.info('A total of {} unique structures where sampled.'.format(len(run_history.data)))
logging.info('A total of {} runs where executed.'.format(len(run_history.get_all_runs())))

y_pred = pipeline.predict(X_test)
y_prob = pipeline.predict_proba(X_test)

logging.info('Final pipeline:\n{}'.format(pipeline))
logging.info('Final test performance {}'.format(util.score(y_test, y_prob, y_pred, ds.metric)))
logging.info('Final ensemble performance {} based on {} individuals'.format(
    util.score(ds_test.y, ensemble.predict_proba(ds_test.X), ensemble.predict(ds_test.X), ds.metric),
    len(ensemble.estimators_)))
