"""
Example 1 - Single Threaded
================================

"""
import argparse
import logging
import sys
from collections import OrderedDict

from sklearn import datasets

# Configure logging system before importing smac
from automl.components.classification.decision_tree import DecisionTree
from dswizard.core.model import Dataset

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)-8s %(name)-20s %(message)s',
                    datefmt='%Y-%m-%dT%H:%M:%S%z',
                    stream=sys.stdout)

from automl.components.classification import ClassifierChoice
from automl.components.data_preprocessing import DataPreprocessorChoice
from dswizard.core.logger import JsonResultLogger
from dswizard.core.master import Master
from dswizard.optimizers.bandit_learners import HyperbandLearner
from dswizard.optimizers.config_generators.hyperopt import Hyperopt
from dswizard.optimizers.structure_generators.fixed import FixedStructure
from dswizard.workers.sklearn_worker import SklearnWorker

logging.getLogger('matplotlib').setLevel(logging.WARNING)

parser = argparse.ArgumentParser(description='Example 1 - sequential and local execution.')
parser.add_argument('--min_budget', type=float, help='Minimum budget used during the optimization.', default=0.01)
parser.add_argument('--max_budget', type=float, help='Maximum budget used during the optimization.', default=1)
parser.add_argument('--timeout', type=float, help='Maximum timeout for a single evaluation in seconds', default=60)
parser.add_argument('--run_id', type=str, help='Name of the run', default='run')
parser.add_argument('--log_dir', type=str, help='Directory used for logging', default='../logs/')
args = parser.parse_args()

# Start worker
X, y = datasets.load_digits(return_X_y=True)
dataset_properties = {
    'target_type': 'classification'
}

# Instantiate optimizer
# sub_wf_2 = OrderedDict()
# sub_wf_2['1.1'] = DataPreprocessorChoice()
# sub_wf_2['1.2'] = ClassifierChoice()

steps = OrderedDict()
# steps['1'] = SubPipeline([sub_wf_2], dataset_properties=dataset_properties)
steps['2'] = DecisionTree()

structure_generator = FixedStructure(steps, dataset_properties, timeout=args.timeout)

master = Master(
    run_id=args.run_id,
    result_logger=JsonResultLogger(directory=args.log_dir, overwrite=True),
    n_workers=1,

    config_generator_class=Hyperopt,

    bandit_learner_class=HyperbandLearner,
    bandit_learner_kwargs={'structure_generator': structure_generator,
                           'min_budget': args.min_budget,
                           'max_budget': args.max_budget}
)

ds = Dataset(X, y, dataset_properties=dataset_properties, test_size=0.3)
try:
    res = master.optimize(ds)

    # Analysis
    id2config = res.get_id2config_mapping()
    incumbent = id2config[res.get_incumbent_id()]

    print('Best found configuration: {} with loss {}'.format(incumbent.get_incumbent().config,
                                                             incumbent.get_incumbent().loss))
    print('A total of {} unique configurations where sampled.'.format(len(id2config.keys())))
    print('A total of {} runs where executed.'.format(len(res.get_all_runs())))

finally:
    master.shutdown()
