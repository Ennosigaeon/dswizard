"""
Example 2 - Sklearn
================================

"""
import argparse
import logging
import sys
from collections import OrderedDict

from sklearn import datasets

import dswizard.core.nameserver as hpns
from dswizard.components.classification import ClassifierChoice
from dswizard.components.data_preprocessing import DataPreprocessorChoice
from dswizard.components.pipeline import SubPipeline
from dswizard.core.master import Master
from dswizard.core.runhistory import JsonResultLogger
from dswizard.optimizers.bandit_learners import GenericBanditLearner
from dswizard.optimizers.structure_generators.fixed import FixedStructure
from dswizard.workers.sklearn_worker import SklearnWorker

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)-8s %(name)-20s %(message)s',
                    datefmt='%Y-%m-%dT%H:%M:%S%z',
                    stream=sys.stdout)
logging.getLogger('Pyro4.core').setLevel(logging.WARNING)

parser = argparse.ArgumentParser(description='Example 1 - sequential and local execution.')
parser.add_argument('--min_budget', type=float, help='Minimum budget used during the optimization.', default=0.01)
parser.add_argument('--max_budget', type=float, help='Maximum budget used during the optimization.', default=1)
parser.add_argument('--timeout', type=float, help='Maximum timeout for a single evaluation', default=5)
parser.add_argument('--run_id', type=str, help='Name of the run', default='run')
parser.add_argument('--log_dir', type=str, help='Directory used for logging', default='../logs/')
args = parser.parse_args()

# Start Nameserver
NS = hpns.NameServer(run_id=args.run_id, host='127.0.0.1', port=None)
NS.start()

# Start worker
X, y = datasets.load_iris(True)
dataset_properties = {
    'target_type': 'classification'
}

w = SklearnWorker(sleep_interval=0, nameserver='127.0.0.1', run_id=args.run_id)
w.set_dataset(X, y, dataset_properties=dataset_properties, test_size=0.3)
w.run(background=True)

# Instantiate optimizer
sub_wf_1 = OrderedDict()
sub_wf_1['sub_step_0'] = ClassifierChoice()

sub_wf_2 = OrderedDict()
sub_wf_2['sub_step_0'] = DataPreprocessorChoice()
sub_wf_2['sub_step_1'] = ClassifierChoice()

steps = OrderedDict()
steps['step_0'] = SubPipeline([sub_wf_1, sub_wf_2], dataset_properties=dataset_properties)
steps['step_1'] = ClassifierChoice()
structure_generator = FixedStructure(dataset_properties, steps)

bandit = GenericBanditLearner(structure_generator,
                              min_budget=args.min_budget,
                              max_budget=args.max_budget,
                              timeout=args.timeout)

master = Master(
    bandit_learner=bandit,
    run_id=args.run_id,
    nameserver='127.0.0.1',
    result_logger=JsonResultLogger(directory=args.log_dir, overwrite=True)
)
res = master.run()

# Shutdown
master.shutdown(shutdown_workers=True)
NS.shutdown()

# Analysis
id2config = res.get_id2config_mapping()
incumbent = id2config[res.get_incumbent_id()]

print('Best found configuration: {} with loss {}'.format(incumbent.get_incumbent().config,
                                                         incumbent.get_incumbent().loss))
print('A total of {} unique configurations where sampled.'.format(len(id2config.keys())))
print('A total of {} runs where executed.'.format(len(res.get_all_runs())))
