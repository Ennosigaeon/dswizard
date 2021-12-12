import os
import pickle

from sklearn.datasets import load_breast_cancer

from dswizard.core.logger import ResultLogger
from dswizard.util.auto_sklearn import load_auto_sklearn_runhistory

X, y = load_breast_cancer(return_X_y=True, as_frame=True)

base_dir = '/home/marc/phd/code/dswizard/scripts/run/autosklearn'

with open(os.path.join(base_dir, 'input', 'auto-sklearn.pkl'), 'rb') as f:
    automl = pickle.load(f)

rh = load_auto_sklearn_runhistory(automl, X.to_numpy(), y.to_numpy(), X.columns,
                                  os.path.join(base_dir, 'input', 'autosklearn_classification_example_tmp'))
logger = ResultLogger(os.path.join(base_dir, 'output'), '.')
logger.log_run_history(rh, suffix='auto-sklearn')

y_pred = automl.predict(X.to_numpy())
