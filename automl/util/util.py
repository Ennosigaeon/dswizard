import numpy as np

from dswizard.components import util


# Only necessary for backwards compatibility of pretrained meta-learning models
def object_log(X: np.ndarray):
    return util.object_log(X)
