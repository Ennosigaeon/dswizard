import importlib
import logging
import os
from typing import Optional

import multiprocessing_logging
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, log_loss
from sklearn.utils.multiclass import type_of_target

valid_metrics = {'accuracy', 'precision', 'recall', 'f1', 'logloss', 'rocauc'}


def setup_logging(log_file: str = None):
    multiprocessing_logging.install_mp_handler()
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(name)-15s %(threadName)-10s %(message)s')

    if log_file is not None:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)

        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def score(y, y_prob, y_pred, metric: str):
    # Always compute minimization problem
    sign = -1

    if metric == 'accuracy':
        score = accuracy_score(y, y_pred)
    elif metric == 'precision':
        score = precision_score(y, y_pred, average='weighted')
    elif metric == 'recall':
        score = recall_score(y, y_pred, average='weighted')
    elif metric == 'f1':
        score = f1_score(y, y_pred, average='weighted')
    elif metric == 'logloss':
        sign = 1
        score = log_loss(y, y_prob)
    elif metric == 'rocauc':
        y_type = type_of_target(y)
        if y_type == "binary" and y_prob.ndim > 1:
            y_prob = y_prob[:, 1]
        score = roc_auc_score(y, y_prob, average='weighted', multi_class='ovr')
    else:
        raise ValueError('Unknown metric {}'.format(metric))

    return sign * score


def worst_score(metric: str):
    if metric in ('accuracy', 'precision', 'recall', 'f1', 'rocauc'):
        return [0, 0]
    else:
        # TODO replace with -log(1 / n_classes)
        # logloss is only used in combination with rocauc
        return [100, 0]


def openml_mapping(task: int = None, ds: int = None, name: str = None):
    tasks = {3: 3, 12: 12, 18: 18, 31: 31, 53: 54, 3549: 458, 3560: 469, 3567: 478, 3896: 1043, 3913: 1063, 7592: 1590,
             9952: 1489, 9961: 1498, 9977: 1486, 9983: 1471, 9986: 1476, 10101: 1464, 14965: 1461, 146195: 40668,
             146212: 40685, 146606: 23512, 146818: 40981, 146821: 40975, 146822: 40984, 167119: 41027,
             167120: 23517, 168329: 41169, 168330: 41168, 168911: 41143, 168912: 41146}
    datasets = dict(map(reversed, tasks.items()))
    names = {'eeg-eye-state': 9983, 'mfeat-morphological': 18, 'segment': 146822, 'sylvine': 168912, 'vehicle': 53,
             'ada_agnostic': 3896, 'adult': 7592, 'analcatdata_authorship': 3549, 'analcatdata_dmft': 3560,
             'Australian': 146818, 'bank-marketing': 14965, 'blood-transfusion': 10101, 'car': 146821, 'collins': 3567,
             'connect-4': 146195, 'credit-g': 31, 'Helena': 168329, 'higgs': 146606, 'Jannis': 168330,
             'jasmine': 168911, 'jungle_chess_2pcs_raw_endgame_complete': 167119, 'kc2': 3913, 'kr-vs-kp': 3,
             'mfeat-factors': 12, 'nomao': 9977, 'numerai28.6': 167120, 'phoneme': 9952, 'sa-heart': 9961,
             'Shuttle': 146212}

    try:
        if task is not None:
            return tasks[task]
        if name is not None:
            return names[name]
        return datasets[ds]
    except KeyError:
        return -1


def prefixed_name(prefix: Optional[str], name: str) -> str:
    """
    Returns the potentially prefixed name name.
    """
    return name if prefix is None else '{}:{}'.format(prefix, name)


def get_type(clazz: str) -> type:
    module_name = clazz.rpartition(".")[0]
    class_name = clazz.split(".")[-1]

    module = importlib.import_module(module_name)
    class_ = getattr(module, class_name)
    return class_


def get_object(clazz: str, kwargs=None):
    if kwargs is None:
        kwargs = {}

    return get_type(clazz)(**kwargs)
