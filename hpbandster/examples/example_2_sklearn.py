"""
Example 2 - Sklearn
================================

"""
import argparse
import logging
import sys
from collections import OrderedDict

from sklearn import datasets

import hpbandster.core.nameserver as hpns
from components.classification import ClassifierChoice
from components.data_preprocessing import DataPreprocessorChoice
from components.pipeline import SubPipeline
from hpbandster.optimizers import BOHB as BOHB
from hpbandster.workers.SklearnWorker import SklearnWorker
from optimizers.structure_generators.fixed import FixedStructure

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%dT%H:%M:%S%z',
                    stream=sys.stdout)
logging.getLogger('Pyro4.core').setLevel(logging.WARNING)

parser = argparse.ArgumentParser(description='Example 1 - sequential and local execution.')
parser.add_argument('--min_budget', type=float, help='Minimum budget used during the optimization.', default=0.01)
parser.add_argument('--max_budget', type=float, help='Maximum budget used during the optimization.', default=1)
parser.add_argument('--n_iterations', type=int, help='Number of iterations performed by the optimizer', default=4)
args = parser.parse_args()

# Start Nameserver
NS = hpns.NameServer(run_id='example1', host='127.0.0.1', port=None)
NS.start()

# Start worker
X, y = datasets.load_iris(True)
dataset_properties = {
    'target_type': 'classification'
}

w = SklearnWorker(sleep_interval=0, nameserver='127.0.0.1', run_id='example1')
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

bohb = BOHB(structure=structure_generator,
            run_id='example1', nameserver='127.0.0.1',
            min_budget=args.min_budget, max_budget=args.max_budget
            )
res = bohb.run(n_iterations=args.n_iterations)

# Shutdown
bohb.shutdown(shutdown_workers=True)
NS.shutdown()

# Analysis
id2config = res.get_id2config_mapping()
incumbent = res.get_incumbent_id()

print('Best found configuration:', id2config[incumbent]['config'])
print('A total of {} unique configurations where sampled.'.format(len(id2config.keys())))
print('A total of {} runs where executed.'.format(len(res.get_all_runs())))
print('Total budget corresponds to {:.1f} full function evaluations.'.format(
    sum([r.budget for r in res.get_all_runs()]) / args.max_budget))
