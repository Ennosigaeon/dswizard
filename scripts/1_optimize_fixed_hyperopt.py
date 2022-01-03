"""
Example 1 - Single Threaded
================================

"""
import argparse
import logging
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from dswizard.components.classification.random_forest import RandomForest
from dswizard.components.classification.decision_tree import DecisionTree
from dswizard.components.classification.libsvm_svc import LibSVM_SVC
from dswizard.components.data_preprocessing.standard_scaler import StandardScalerComponent
from dswizard.components.feature_preprocessing.one_hot_encoding import OneHotEncoderComponent
from dswizard.components.sklearn import ColumnTransformerComponent
from dswizard.core.master import Master
from dswizard.core.model import Dataset
from dswizard.optimizers.bandit_learners.pseudo import PseudoBandit
from dswizard.util import util
from optimizers.config_generators import Hyperopt
from optimizers.structure_generators.fixed import FixedStructure

parser = argparse.ArgumentParser(description='Example 1 - dswizard optimization.')
parser.add_argument('--wallclock_limit', type=float, help='Maximum optimization time for in seconds', default=60)
parser.add_argument('--cutoff', type=float, help='Maximum cutoff time for a single evaluation in seconds', default=-1)
parser.add_argument('--log_dir', type=str, help='Directory used for logging', default='run/')
parser.add_argument('--fold', type=int, help='Fold of OpenML task to optimize', default=0)

task = -2
args = parser.parse_args()

util.setup_logging(os.path.join(args.log_dir, str(task), 'log.txt'))
logger = logging.getLogger()
logging.getLogger('matplotlib').setLevel(logging.WARNING)

data = pd.read_csv('/home/marc/phd/code/xautoml/xautoml/tests/res/autosklearn_hearts/dataset.csv')
X = data.loc[:, data.columns[:-1]]
y = data.loc[:, data.columns[-1]]

X.loc[:, 'Sex'] = X.Sex.astype('category')
X.loc[:, 'ChestPainType'] = X.ChestPainType.astype('category')
X.loc[:, 'RestingECG'] = X.RestingECG.astype('category')
X.loc[:, 'ExerciseAngina'] = X.ExerciseAngina.astype('category')
X.loc[:, 'ST_Slope'] = X.ST_Slope.astype('category')

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
ds = Dataset(X_train.values, y_train.values, task=-2, metric='accuracy', feature_names=X_train.columns)
ds_test = Dataset(X_test.values, y_test.values, task=-2, metric='accuracy', feature_names=X_test.columns)

ds, ds_test = Dataset.from_openml(3937, 0, 'accuracy')

steps = [
    ('data_preprocessing', ColumnTransformerComponent(
        # [('onehot', OneHotEncoderComponent(), [1, 2, 6, 8, 10]),
        #  ('scaler', StandardScalerComponent(), [0, 3, 4, 5, 7, 9])],
        [('onehot', OneHotEncoderComponent(), [0, 3, 4, 5]),
         ('scaler', StandardScalerComponent(), [1, 2, 6, 7, 8, 9])],
        remainder='passthrough')),
    ('svm', RandomForest())
]

master = Master(
    ds=ds,
    working_directory=os.path.join(args.log_dir, str(task)),
    n_workers=1,
    model='../dswizard/assets/rf_complete.pkl',

    wallclock_limit=args.wallclock_limit,
    cutoff=args.cutoff,
    pre_sample=False,

    config_generator_class=Hyperopt,

    structure_generator_class=FixedStructure,
    structure_generator_kwargs={'steps': steps},

    bandit_learner_class=PseudoBandit
)

pipeline, run_history, ensemble = master.optimize()

# Analysis
_, incumbent = run_history.get_incumbent()

logging.info(f'Best found configuration: {incumbent.steps}\n'
             f'{incumbent.get_incumbent().config} with loss {incumbent.get_incumbent().loss}')
logging.info(f'A total of {len(run_history.data)} unique structures where sampled.')
logging.info(f'A total of {len(run_history.get_all_runs())} runs where executed.')

y_pred = pipeline.predict(ds_test.X)
y_prob = pipeline.predict_proba(ds_test.X)

logging.info(f'Final pipeline:\n{pipeline}')
logging.info(f'Final test performance {util.score(ds_test.y, y_prob, y_pred, ds.metric)}')
logging.info(f'Final ensemble performance '
             f'{util.score(ds_test.y, ensemble.predict_proba(ds_test.X), ensemble.predict(ds_test.X), ds.metric)} '
             f'based on {len(ensemble.estimators_)} individuals')
