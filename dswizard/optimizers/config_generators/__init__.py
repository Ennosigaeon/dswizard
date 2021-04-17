import warnings

from dswizard.optimizers.config_generators.random_sampling import RandomSampling
from dswizard.optimizers.config_generators.hyperopt import Hyperopt

try:
    from dswizard.optimizers.config_generators.smac_generator import SmacGenerator
except ImportError:
    warnings.warn("SMAC not installed")
