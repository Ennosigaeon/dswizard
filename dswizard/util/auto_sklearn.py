import os
from collections import OrderedDict
from typing import Union

import numpy as np
import pandas as pd
from ConfigSpace import ConfigurationSpace
from autosklearn.estimators import AutoSklearnEstimator
from autosklearn.pipeline.components.data_preprocessing import DataPreprocessorChoice
from sklearn.pipeline import Pipeline
from smac.scenario.scenario import Scenario
from smac.stats.stats import Stats

from dswizard.components.sklearn import ColumnTransformerComponent
from dswizard.core.model import Dataset
from dswizard.core.model import MetaInformation, Result, CandidateId, Runtime, CandidateStructure, StatusType
from dswizard.core.runhistory import RunHistory
from dswizard.pipeline.pipeline import FlexiblePipeline


def load_auto_sklearn_runhistory(automl: AutoSklearnEstimator,
                                 X: Union[np.ndarray, pd.DataFrame],
                                 y: Union[np.ndarray, pd.DataFrame],
                                 feature_names: list[str],
                                 dir: str = None):
    def build_meta_information() -> MetaInformation:
        try:
            start_time = backend.load_start_time(automl.seed)
        except FileNotFoundError:
            start_time = next(iter(runhistory.data.values())).starttime

        arguments = {
            'tmp_folder': backend.context._temporary_directory,
            'time_left_for_this_task': automl.automl_._time_for_task,
            'per_run_time_limit': automl.automl_._per_run_time_limit,
            'ensemble_size': automl.automl_._ensemble_size,
            'ensemble_nbest': automl.automl_._ensemble_nbest,
            'max_models_on_disc': automl.automl_._max_models_on_disc,
            'seed': automl.automl_._seed,
            'memory_limit': automl.automl_._memory_limit,
            'metadata_directory': automl.automl_._metadata_directory,
            'debug_mode': automl.automl_._debug_mode,
            'include': automl.automl_._include,
            'exclude': automl.automl_._exclude,
            'resampling_strategy': automl.automl_._resampling_strategy,
            'resampling_strategy_arguments': automl.automl_._resampling_strategy_arguments,
            'n_jobs': automl.automl_._n_jobs,
            'multiprocessing_context': automl.automl_._multiprocessing_context,
            'dask_client': automl.automl_._dask_client,
            'precision': automl.automl_.precision,
            'disable_evaluator_output': automl.automl_._disable_evaluator_output,
            'get_smac_objective_callback': automl.automl_._get_smac_object_callback,
            'smac_scenario_args': automl.automl_._smac_scenario_args,
            'logging_config': automl.automl_.logging_config,
        }

        scenario = Scenario({"run_obj": "quality"})
        scenario.output_dir_for_this_run = backend.get_smac_output_directory_for_run(automl.seed)
        stats = Stats(scenario)
        stats.load()

        meta = MetaInformation(start_time,
                               metric.name,
                               None,
                               None,
                               data_file,
                               arguments
                               )
        meta.end_time = start_time + stats.wallclock_time_used
        meta.n_configs = stats.n_configs
        meta.n_structures = 1
        meta.iterations = {}
        meta.incumbent = trajectory[-1][0] if meta.is_minimization else metric._optimum - trajectory[-1][0]

        return meta

    def build_structures() -> dict[CandidateId, CandidateStructure]:
        def as_flexible_pipeline(simple_pipeline: Pipeline) -> FlexiblePipeline:
            def suffix_name(name, suffix) -> str:
                if suffix is None:
                    return name
                elif name in suffix:
                    return name
                else:
                    return name
                    # return '{}:{}'.format(name, suffix)

            def convert_component(component):
                suffix = component.__class__.__module__.split('.')[-1]

                if hasattr(component, 'choice'):
                    return convert_component(component.choice)

                if hasattr(component, 'column_transformer'):
                    c, suffix2 = convert_component(component.column_transformer)
                    return c, suffix

                if hasattr(component, 'transformers'):
                    transformers = []
                    for name, t, cols in component.transformers:
                        t_, suffix = convert_component(t)
                        transformers.append((suffix_name(name, suffix), t_, cols))

                    return ColumnTransformerComponent(transformers), suffix

                if hasattr(component, 'steps'):
                    steps = []
                    for name, s in component.steps:
                        if isinstance(s, DataPreprocessorChoice):
                            name = '{}:{}'.format(name, s.choice.__class__.__module__.split('.')[-1])

                        s_, suffix = convert_component(s)
                        steps.append((suffix_name(name, suffix), s_))
                    return FlexiblePipeline(steps), None

                return component, suffix

            a = convert_component(simple_pipeline)[0]
            return a

        structures = OrderedDict()
        for key, value in runhistory.data.items():
            try:
                pipeline = backend.load_model_by_seed_and_id_and_budget(automl.seed, key.config_id, key.budget)
                flexible_pipeline = as_flexible_pipeline(pipeline)
            except FileNotFoundError:
                # Skip results without fitted model for now
                # TODO also load failed configurations
                continue

            try:
                struct_key = str(flexible_pipeline.get_hyperparameter_search_space())
                cs = None
            except AttributeError:
                # Fallback for DummyClassifier
                cs = ConfigurationSpace()
                struct_key = str(cs)

            if struct_key not in structures:
                structure = CandidateStructure(cs, flexible_pipeline, [], )
                structure.cid = CandidateId(0, len(structures))
                structures[struct_key] = structure
            else:
                structure = structures[struct_key]

            if key.config_id > 1:
                config = automl.automl_.runhistory_.ids_config.get(key.config_id - 1)
            else:
                config = cs.get_default_configuration()
                config.origin = 'Default'

            result = Result(
                structure.cid.with_config(key.config_id),
                StatusType.SUCCESS if value.status.name == 'SUCCESS' else StatusType.CRASHED,
                config,
                value.cost if metric._sign != 1.0 else -metric._optimum + value.cost,
                None,
                Runtime(value.endtime - value.starttime, value.starttime - meta_information.start_time)
            )
            result.model_file = os.path.abspath(
                os.path.join(backend.get_numrun_directory(automl.seed, key.config_id, key.budget),
                             backend.get_model_filename(automl.seed, key.config_id, key.budget))
            )
            structure.results.append(result)

        return {s.cid: s for s in structures.values()}

    metric = automl.automl_._metric
    backend = automl.automl_._backend
    runhistory = automl.automl_.runhistory_
    trajectory = automl.trajectory_

    if dir is not None:
        automl.tmp_folder = dir
        automl.automl_.temporary_directory = dir
        backend.context._temporary_directory = dir
        backend.internals_directory = os.path.join(dir, ".auto-sklearn")

    # auto-sklearn does not store trainings data
    data_file = os.path.join(automl.tmp_folder, 'dataset.pkl')
    _, y = automl.automl_.InputValidator.transform(X, y)
    Dataset(X, y, feature_names=feature_names).store(data_file)

    meta_information = build_meta_information()
    structures = build_structures()

    return RunHistory(structures, meta_information, {}, automl.automl_.configuration_space)
