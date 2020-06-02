"""
Example 1 - Single Threaded
================================

"""
import argparse
import logging
import sys

import sklearn
from sklearn import datasets

from automl.components.classification.decision_tree import DecisionTree
from automl.components.data_preprocessing import DataPreprocessorChoice
from automl.components.feature_preprocessing import FeaturePreprocessorChoice
from dswizard.core.logger import JsonResultLogger
from dswizard.core.master import Master
from dswizard.core.model import Dataset
from dswizard.optimizers.bandit_learners import HyperbandLearner
from dswizard.optimizers.structure_generators.fixed import FixedStructure
from optimizers.config_generators import Hyperopt
from optimizers.structure_generators.mcts import MCTS

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)-8s %(name)-20s %(message)s',
                    datefmt='%Y-%m-%dT%H:%M:%S%z',
                    stream=sys.stdout)

logging.getLogger('matplotlib').setLevel(logging.WARNING)

parser = argparse.ArgumentParser(description='Example 1 - sequential and local execution.')
parser.add_argument('--min_budget', type=float, help='Minimum budget used during the optimization.', default=1)
parser.add_argument('--max_budget', type=float, help='Maximum budget used during the optimization.', default=10)
parser.add_argument('--n_configs', type=float, help='Number of configurations to test on a single structure', default=1)
parser.add_argument('--wallclock_limit', type=float, help='Maximum optimization time for in seconds', default=60)
parser.add_argument('--cutoff', type=float, help='Maximum cutoff time for a single evaluation in seconds', default=60)
parser.add_argument('--run_id', type=str, help='Name of the run', default='run')
parser.add_argument('--log_dir', type=str, help='Directory used for logging', default='run/logs/')
args = parser.parse_args()

# Load dataset
X, y = datasets.load_digits(return_X_y=True)
X, y = sklearn.utils.shuffle(X, y)
ds = Dataset(X, y)

steps = [
    ('1', DataPreprocessorChoice()),
    # ('2', FeaturePreprocessorChoice()),
    ('3', DecisionTree())
]

master = Master(
    ds=ds,
    run_id=args.run_id,
    result_logger=JsonResultLogger(directory=args.log_dir, overwrite=True),
    n_workers=1,

    wallclock_limit=args.wallclock_limit,
    cutoff=args.cutoff,
    pre_sample=False,

    config_generator_class=Hyperopt,

    structure_generator_class=FixedStructure,
    structure_generator_kwargs={'steps': steps},

    bandit_learner_class=HyperbandLearner,
    bandit_learner_kwargs={'min_budget': args.min_budget,
                           'max_budget': args.max_budget}
)

try:
    res = master.optimize()

    # Analysis
    id2config = res.get_id2config_mapping()
    incumbent = id2config[res.get_incumbent_id()]

    print('Best found configuration: {}\n{} with loss {}'.format(incumbent.get_incumbent().steps,
                                                                 incumbent.get_incumbent().config,
                                                                 incumbent.get_incumbent().loss))
    print('A total of {} structures where sampled.'.format(len(id2config.keys())))
    print('A total of {} runs where executed.'.format(len(res.get_all_runs())))

finally:
    master.shutdown()
