"""
Example 1 - Single Threaded
================================

"""
import argparse
import logging
import os

import pandas as pd
from sklearn.model_selection import train_test_split

from dswizard.components.classification.decision_tree import DecisionTree
from dswizard.components.data_preprocessing.imputation import ImputationComponent
from dswizard.components.data_preprocessing.minmax import MinMaxScalerComponent
from dswizard.components.feature_preprocessing.ordinal_encoder import OrdinalEncoderComponent
from dswizard.components.feature_preprocessing.pca import PCAComponent
from dswizard.components.sklearn import ColumnTransformerComponent
from dswizard.components.sklearn import FeatureUnionComponent
from dswizard.core.master import Master
from dswizard.core.model import Dataset
from dswizard.optimizers.bandit_learners.pseudo import PseudoBandit
from dswizard.optimizers.config_generators import SmacGenerator
from dswizard.optimizers.structure_generators.fixed import FixedStructure
from dswizard.pipeline.pipeline import FlexiblePipeline
from dswizard.util import util
from optimizers.config_generators import Hyperopt
from optimizers.structure_generators.mcts import MCTS, TransferLearning

parser = argparse.ArgumentParser(description='Example 1 - dswizard optimization.')
parser.add_argument('--wallclock_limit', type=float, help='Maximum optimization time for in seconds', default=600)
parser.add_argument('--cutoff', type=float, help='Maximum cutoff time for a single evaluation in seconds', default=10)
parser.add_argument('--log_dir', type=str, help='Directory used for logging', default='run/')
parser.add_argument('--fold', type=int, help='Fold of OpenML task to optimize', default=0)
parser.add_argument('task', type=int, help='OpenML task id')
args = parser.parse_args()

util.setup_logging(os.path.join(args.log_dir, str(args.task), 'log.txt'))
logger = logging.getLogger()
logging.getLogger('matplotlib').setLevel(logging.WARNING)

# Load dataset
# Tasks: 18, 53, 9983, 146822, 168912
logger.info(f'Processing task {args.task}')
# ds, ds_test = Dataset.from_openml(args.task, args.fold, 'roc_auc')


data = pd.read_csv('/home/marc/phd/code/xautoml/xautoml/tests/res/autosklearn_hearts/dataset.csv')
X = data.loc[:, data.columns[:-1]]
y = data.loc[:, data.columns[-1]]

X.loc[:, 'Sex'] = X.Sex.astype('category')
X.loc[:, 'ChestPainType'] = X.ChestPainType.astype('category')
X.loc[:, 'RestingECG'] = X.RestingECG.astype('category')
X.loc[:, 'ExerciseAngina'] = X.ExerciseAngina.astype('category')
X.loc[:, 'ST_Slope'] = X.ST_Slope.astype('category')

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=1)
ds = Dataset(X_train.values, y_train.values, metric='roc_auc', feature_names=X_train.columns)
ds_test = Dataset(X_test.values, y_test.values, metric='roc_auc', feature_names=X_test.columns)

# 168746
# steps = [
#     ('data_preprocessing', ColumnTransformerComponent([
#         ("imputation", ImputationComponent(), [0, 3, 4, 5, 7, 11]),
#         ("categorical",
#          FlexiblePipeline([('ordinal_encoder', OrdinalEncoderComponent()), ('imputation', ImputationComponent())]),
#          [1, 2, 6, 8, 9, 10, 12]),
#     ])),
#     ('parallel', FeatureUnionComponent([('pca', PCAComponent()), ('minmax_scaler', MinMaxScalerComponent())])),
#     ('decision_tree', DecisionTree())
# ]

# pip = FlexiblePipeline(steps)
# pip.set_hyperparameters(pip.get_hyperparameter_search_space().get_default_configuration())
# pip.fit(ds.X, ds.y)
# print(pip.get_feature_names_out(ds.feature_names))

# 59
# steps = [
#     ('1', FeatureUnionComponent([
#         ('1.1', FlexiblePipeline([('1.1.1', KNNImputerComponent()), ('1.1.2', MinMaxScalerComponent())])),
#         ('1.2', PCAComponent())
#     ])),
#     ('2', DecisionTree())
# ]

master = Master(
    ds=ds,
    working_directory=os.path.join(args.log_dir, str(args.task)),
    n_workers=1,
    model='../dswizard/assets/rf_complete.pkl',

    wallclock_limit=args.wallclock_limit,
    cutoff=args.cutoff,
    pre_sample=False,

    config_generator_class=Hyperopt,

    structure_generator_class=MCTS,
    structure_generator_kwargs={'policy': TransferLearning},
    # structure_generator_class=FixedStructure,
    # structure_generator_kwargs={'steps': steps},

    bandit_learner_class=PseudoBandit
)

pipeline, run_history, ensemble = master.optimize()
print(pipeline.get_feature_names_out(ds.feature_names))

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
