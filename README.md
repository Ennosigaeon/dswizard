# dswizard

_dswizard_ is an efficient solver for machine learning (ML) pipeline synthesis inspired by human behaviour. It
automatically derives a pipeline structure, selects algorithms and performs hyperparameter optimization. This repository
contains the source code and data used in our publication [Iterative Search Space Construction for Pipeline Structure Search](https://arxiv.org/).

## How to install

The code has only be tested with Python 3.8, but any version supporting type hints should work. We recommend using a
virtual environment.
```
python3 -m virtualenv venv
source venv/bin/activate
```

_dswizard_ is available on PyPI, you can simply install it via
```
pip install dswizard
```

Alternatively, you can checkout the source code and install it directly via
```
pip install -e dswizard
```

Now you are ready to go.

### Visualization
`dswizard` contains an optional pipeline search space visualization functionality intended for debugging and
explainability. If you don't need this feature, you can skip this step. To use the visualization you have to install
[Graphviz](https://graphviz.org/) manually and add the additional visualization libraries using
```
pip install dswizard[visualization]
```


## Usage

In the folder scripts, we have provided scripts to showcase usage of _dswizard_. The most important script is
`scripts/1_optimize.py`. This script solves the pipeline synthesis for a given task. To get usage information use
```
python dswizard/scripts/1_optimize.py --help
```
yielding a output similar to

    usage: 1_optimize.py [-h] [--wallclock_limit WALLCLOCK_LIMIT] [--cutoff CUTOFF] [--log_dir LOG_DIR] task
    
    Example 1 - dswizard optimization.
    
    positional arguments:
      task                  OpenML task id
    
    optional arguments:
      -h, --help            show this help message and exit
      --wallclock_limit WALLCLOCK_LIMIT
                            Maximum optimization time for in seconds
      --cutoff CUTOFF       Maximum cutoff time for a single evaluation in seconds
      --log_dir LOG_DIR     Directory used for logging
      --fold FOLD           Fold of OpenML task to optimize


You have to pass an [OpenML](https://www.openml.org/) task id. For example, to create pipelines for the _kc2_ data set
use `python dswizard/scripts/1_optimize.py 3913`. Via the optional parameter you can change the total optimization time
(default 300 seconds), maximum evaluation time for a single configuration (default 60 seconds), the directory to store
optimization artifacts (default _run/{TASK}_) and the fold to evaluate (default 0).

The optimization procedure prints the best found pipeline structure with the according configuration and test performance
to the console., similar to

    2020-11-13 16:45:55,312 INFO     root            MainThread Best found configuration: [('22', KBinsDiscretizer), ('26', PCAComponent), ('28', AdaBoostingClassifier)]
    Configuration:
      22:encode, Value: 'ordinal'
      22:n_bins, Value: 32
      22:strategy, Value: 'kmeans'
      26:keep_variance, Value: 0.9145797030897109
      26:whiten, Value: True
      28:algorithm, Value: 'SAMME'
      28:learning_rate, Value: 0.039407336108331845
      28:n_estimators, Value: 138
     with loss -0.8401893431635389
    2020-11-13 16:45:55,312 INFO     root            MainThread A total of 20 unique structures where sampled.
    2020-11-13 16:45:55,312 INFO     root            MainThread A total of 58 runs where executed.
    2020-11-13 16:45:55,316 INFO     root            MainThread Final pipeline:
    FlexiblePipeline(configuration={'22:encode': 'ordinal', '22:n_bins': 32,
                                    '22:strategy': 'kmeans',
                                    '26:keep_variance': 0.9145797030897109,
                                    '26:whiten': True, '28:algorithm': 'SAMME',
                                    '28:learning_rate': 0.039407336108331845,
                                    '28:n_estimators': 138},
                     steps=[('22', KBinsDiscretizer(encode='ordinal', n_bins=32, strategy='kmeans')),
                            ('26', PCAComponent(keep_variance=0.9145797030897109, whiten=True)),
                            ('28', AdaBoostingClassifier(algorithm='SAMME', learning_rate=0.039407336108331845, n_estimators=138))])
    2020-11-13 16:45:55,828 INFO     root            MainThread Final test performance -0.8430735930735931

Additionally, an ensemble of the evaluated pipeline candidates is constructed.

    2020-11-13 16:46:06,371 DEBUG    Ensemble        MainThread Building bagged ensemble
    2020-11-13 16:46:09,606 DEBUG    Ensemble        MainThread Ensemble constructed
    2020-11-13 16:46:10,472 INFO     root            MainThread Final ensemble performance -0.8528138528138528 based on 11 pipelines

In the log directory (default _run/{task}_) four files are stored:

1. _log.txt_ contains the complete logging output
2. _results.json_ contains detailed information about all evaluated hyperparameter configurations.
3. _search_graph.pdf_ is a visual representation of the internal pipeline structure graph.
4. _structures.json_ contains all tested pipeline structures including the list of algorithms and the complete configuration space.


## Benchmarking

To assess the performance of _dswizard_ we have implemented an adapter for the OpenML [automlbenchmark](https://github.com/openml/automlbenchmark) available 
[here](https://github.com/Ennosigaeon/automlbenchmark). Please refer to that repository for benchmarking _dswizard_. The
file `scripts/2_load_pipelines.py`, `scripts/3_load_performance.py` and `scripts/4_load_trajectories.py` are used to
compare _dswizard_ with _autosklearn_ and _tpot_, both also evaluated via _automlbenchmark_.


## Meta-Learning

The meta-learning base used in this repository is created using [meta-learning-base](https://github.com/Ennosigaeon/meta-learning-base).
Please see this repository on how to create the required meta-learning models.

For simplicity, we directly provide a random and sgd forest regression model trained on all available data ready to use.
It is available in in `dswizard/assets/`. `scripts/1_optimize.py` is already configured to use this model.

The data used to train the regression model is also available [online](https://github.com/Ennosigaeon/meta-learning-base/tree/master/assets/defaults).
Please refer to [meta-learning-base](https://github.com/Ennosigaeon/meta-learning-base) to see how to train the model
from the raw data.


## dswizard-components

This repository only contains the optimization logic. The actual basic ML components to be optimized are available in
[_dswizard-components_](https://github.com/Ennosigaeon/dswizard-components). Currently, only _sklearn_ components are
supported.


## Release New Version

Increase the version number in `setup.py` and build a new release with `python setup.py sdist`. Finally, upload the
new version using `twine upload dist/dswizard-<VERSION>.tar.gz`.
