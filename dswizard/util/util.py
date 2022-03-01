import logging
import os
from collections import Counter
from typing import Tuple, List

import multiprocessing_logging
from ConfigSpace import Configuration, ConfigurationSpace
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, log_loss
from sklearn.utils.multiclass import type_of_target
from slugify import slugify

from dswizard.components.util import prefixed_name

valid_metrics = {'accuracy', 'precision', 'recall', 'f1', 'logloss', 'roc_auc'}


def setup_logging(log_file: str = None):
    logging.getLogger('matplotlib').setLevel(logging.WARNING)

    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    while len(logger.handlers) > 0:
        logger.removeHandler(logger.handlers[0])

    multiprocessing_logging.install_mp_handler(logger)
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
    if metric == 'accuracy':
        s = accuracy_score(y, y_pred)
    elif metric == 'precision':
        s = precision_score(y, y_pred, average='weighted')
    elif metric == 'recall':
        s = recall_score(y, y_pred, average='weighted')
    elif metric == 'f1':
        s = f1_score(y, y_pred, average='weighted')
    elif metric == 'logloss':
        s = log_loss(y, y_prob)
    elif metric == 'roc_auc':
        y_type = type_of_target(y)
        if y_type == "binary" and y_prob.ndim > 1:
            y_prob = y_prob[:, 1]
        s = roc_auc_score(y, y_prob, average='weighted', multi_class='ovr')
    else:
        raise ValueError(f'Unknown metric {metric}')

    return metric_sign(metric) * s


def metric_sign(metric: str) -> int:
    # Always compute minimization problem
    if metric == 'logloss':
        return 1
    return -1


def worst_score(metric: str) -> Tuple[float, float]:
    if metric in ('accuracy', 'precision', 'recall', 'f1', 'roc_auc'):
        return 0, 0
    else:
        # TODO replace with -log(1 / n_classes)
        # logloss is only used in combination with rocauc
        return 100, 0


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


# No typehint due to circular import with model.py
def model_file(cid) -> str:
    if isinstance(cid.config, str):
        return f'step_{slugify(cid.config)}.pkl'
    else:
        return 'models_{}-{}-{}.pkl'.format(*cid.as_tuple())


def merge_configurations(partial_configs,  # type: List[PartialConfig]
                         cs: ConfigurationSpace) -> Configuration:
    complete = {}
    for partial_config in partial_configs:
        for param, value in partial_config.config.get_dictionary().items():
            param = prefixed_name(partial_config.name, param)
            complete[param] = value

    config = Configuration(cs, complete)
    config.origin = Counter([p.config.origin for p in partial_configs if not p.is_empty()]).most_common(1)[0][0]
    return config
