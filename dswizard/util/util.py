import importlib
from typing import Optional

from sklearn.metrics import roc_auc_score, log_loss
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

valid_metrics = {'accuracy', 'precision', 'recall', 'f1', 'logloss', 'rocauc'}


def score(y, y_pred, metric: str):
    if metric == 'accuracy':
        score = accuracy_score(y, y_pred)
    elif metric == 'precision':
        score = precision_score(y, y_pred, average='weighted')
    elif metric == 'recall':
        score = recall_score(y, y_pred, average='weighted')
    elif metric == 'f1':
        score = f1_score(y, y_pred, average='weighted')
    elif metric == 'logloss':
        # TODO not working
        score = logloss(y, y_pred)
    elif metric == 'rocauc':
        score = multiclass_roc_auc_score(y, y_pred, average='weighted')
    else:
        raise ValueError

    # Always compute minimization problem
    if metric != 'logloss':
        score = -1 * score
    return score


def multiclass_roc_auc_score(y_test, y_pred, average="macro"):
    """
    from https://medium.com/@plog397/auc-roc-curve-scoring-function-for-multi-class-classification-9822871a6659
    """
    lb = LabelBinarizer()
    lb.fit(y_test)

    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)

    return roc_auc_score(y_test, y_pred, average=average)


def logloss(y_test, y_pred):
    """
    from https://medium.com/@plog397/auc-roc-curve-scoring-function-for-multi-class-classification-9822871a6659
    """
    lb = LabelBinarizer()
    lb.fit(y_test)

    y_test = lb.transform(y_test)
    y_pred = lb.transform(y_pred)

    return log_loss(y_test, y_pred)


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
