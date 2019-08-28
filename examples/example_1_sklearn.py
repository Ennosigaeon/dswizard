"""
Example 2 - Sklearn
================================

"""
import argparse
import logging
import sys

from sklearn import datasets

from dswizard.core.config_generator_cache import ConfigGeneratorCache
from dswizard.core.master import Master
from dswizard.core.runhistory import JsonResultLogger
from dswizard.optimizers.bandit_learners import GenericBanditLearner
from dswizard.optimizers.config_generators import Hyperopt
from dswizard.optimizers.structure_generators.random import RandomStructureGenerator
from dswizard.workers.sklearn_worker import SklearnWorker

logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(levelname)-8s %(name)-20s %(message)s',
                    datefmt='%Y-%m-%dT%H:%M:%S%z',
                    stream=sys.stdout)
logging.getLogger('Pyro4.core').setLevel(logging.WARNING)

parser = argparse.ArgumentParser(description='Example 1 - sequential and local execution.')
parser.add_argument('--min_budget', type=float, help='Minimum budget used during the optimization.', default=0.01)
parser.add_argument('--max_budget', type=float, help='Maximum budget used during the optimization.', default=1)
parser.add_argument('--timeout', type=float, help='Maximum timeout for a single evaluation in seconds', default=5)
parser.add_argument('--run_id', type=str, help='Name of the run', default='run')
parser.add_argument('--log_dir', type=str, help='Directory used for logging', default='../logs/')
args = parser.parse_args()

# Start worker
X, y = datasets.load_breast_cancer(True)
dataset_properties = {
    'target_type': 'classification'
}

w = SklearnWorker(run_id=args.run_id, wid='0')
w.set_dataset(X, y, dataset_properties=dataset_properties, test_size=0.3)
w.run(background=True)

# Instantiate optimizer
structure_generator = RandomStructureGenerator(dataset_properties, timeout=args.timeout)
cfg = ConfigGeneratorCache.instance(clazz=Hyperopt, init_args={})
bandit = GenericBanditLearner(structure_generator,
                              min_budget=args.min_budget,
                              max_budget=args.max_budget)

master = Master(
    run_id=args.run_id,
    bandit_learner=bandit,
    result_logger=JsonResultLogger(directory=args.log_dir, overwrite=True),

    local_workers=[w],
)
res = master.run()

# Shutdown
master.shutdown(shutdown_workers=True)

# Analysis
id2config = res.get_id2config_mapping()
incumbent = id2config[res.get_incumbent_id()]

print('Best found configuration: {} with loss {}'.format(incumbent.get_incumbent().config,
                                                         incumbent.get_incumbent().loss))
print('A total of {} unique configurations where sampled.'.format(len(id2config.keys())))
print('A total of {} runs where executed.'.format(len(res.get_all_runs())))
