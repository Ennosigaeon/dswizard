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
from dswizard.optimizers.bandit_learners import HyperbandLearner
from dswizard.optimizers.config_generators import Hyperopt
from dswizard.optimizers.structure_generators.mcts import MCTS, TransferLearning
from dswizard.util import util

parser = argparse.ArgumentParser(description='Example 1 - sequential and local execution.')
parser.add_argument('--min_budget', type=float, help='Minimum budget used during the optimization.', default=1)
parser.add_argument('--max_budget', type=float, help='Maximum budget used during the optimization.', default=10)
parser.add_argument('--n_configs', type=float, help='Number of configurations to test on a single structure', default=1)
parser.add_argument('--wallclock_limit', type=float, help='Maximum optimization time for in seconds', default=3600)
parser.add_argument('--cutoff', type=float, help='Maximum cutoff time for a single evaluation in seconds', default=300)
parser.add_argument('--log_dir', type=str, help='Directory used for logging', default='run/')
parser.add_argument('--task', type=int, help='OpenML task id', default=9983)
args = parser.parse_args()

util.setup_logging(os.path.join(args.log_dir, str(args.task), 'log.txt'))
logger = logging.getLogger()
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# Load dataset
# Tasks: 18, 53, 9983, 146822, 168912
logger.info('Processing task {}'.format(args.task))
task = openml.tasks.get_task(args.task)
train_indices, test_indices = task.get_train_test_split_indices(repeat=0, fold=0, sample=0)

# noinspection PyUnresolvedReferences
X, y, _, _ = task.get_dataset().get_data(task.target_name)
X_train = X.loc[train_indices]
y_train = y[train_indices]
X_test = X.loc[test_indices]
y_test = y[test_indices]

ds = Dataset(X_train.to_numpy(), y_train.to_numpy())

master = Master(
    ds=ds,
    working_directory=os.path.join(args.log_dir, str(args.task)),
    n_workers=4,

    wallclock_limit=args.wallclock_limit,
    cutoff=args.cutoff,
    pre_sample=False,

    config_generator_class=Hyperopt,

    structure_generator_class=MCTS,
    structure_generator_kwargs={'policy': TransferLearning,
                                'policy_kwargs': {'task': args.task, 'dir': '../dswizard/assets'}},

    bandit_learner_class=HyperbandLearner,
    bandit_learner_kwargs={'min_budget': args.min_budget,
                           'max_budget': args.max_budget}
)

try:
    pipeline, run_history = master.optimize()

    # Analysis
    id2config = run_history.get_id2config_mapping()
    _, incumbent = run_history.get_incumbent()

    print('Best found configuration: {}\n{} with loss {}'.format(incumbent.steps,
                                                                 incumbent.get_incumbent().config,
                                                                 incumbent.get_incumbent().loss))
    print('A total of {} structures where sampled.'.format(len(id2config.keys())))
    print('A total of {} runs where executed.'.format(len(run_history.get_all_runs())))

    print('Final pipeline:\n{}'.format(pipeline))
    pipeline.fit(X, y)
    predictions = pipeline.predict(X_test)
    print('Final test performance', util.score(y_test, predictions, ds.metric))

finally:
    master.shutdown()
