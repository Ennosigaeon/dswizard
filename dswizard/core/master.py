from __future__ import annotations

import datetime
import logging
import multiprocessing
import os
import os.path
import random
import tempfile
import threading
import time
import timeit
from multiprocessing.managers import SyncManager
from typing import Type, TYPE_CHECKING, Tuple, Dict, Optional, Union

import joblib
from ConfigSpace.configuration_space import ConfigurationSpace

from dswizard.core.base_structure_generator import BaseStructureGenerator
from dswizard.core.config_cache import ConfigCache
from dswizard.core.constants import MODEL_DIR
from dswizard.core.dispatcher import Dispatcher
from dswizard.core.ensemble import EnsembleBuilder
from dswizard.core.logger import ResultLogger
from dswizard.core.model import StructureJob, Dataset, EvaluationJob, CandidateStructure, CandidateId, MetaInformation
from dswizard.core.renderer import NotebookRenderer
from dswizard.core.runhistory import RunHistory
from dswizard.optimizers.bandit_learners import PseudoBandit
from dswizard.optimizers.config_generators import Hyperopt
from dswizard.optimizers.structure_generators.mcts import MCTS
from dswizard.pipeline.voting_ensemble import PrefitVotingClassifier
from dswizard.workers import SklearnWorker

if TYPE_CHECKING:
    from dswizard.core.base_bandit_learner import BanditLearner
    from dswizard.core.base_config_generator import BaseConfigGenerator
    from dswizard.core.worker import Worker
    from dswizard.pipeline.pipeline import FlexiblePipeline


class Master:
    def __init__(self,
                 ds: Dataset,
                 working_directory: str = '.',
                 model: str = None,
                 logger: logging.Logger = None,
                 result_logger: ResultLogger = None,

                 wallclock_limit: int = 60,
                 cutoff: int = None,
                 structure_cutoff_factor: float = 2.,
                 pre_sample: bool = False,

                 n_workers: int = 1,
                 worker_class: Type[Worker] = SklearnWorker,

                 config_generator_class: Type[BaseConfigGenerator] = Hyperopt,
                 config_generator_kwargs: Dict = None,

                 structure_generator_class: Type[BaseStructureGenerator] = MCTS,
                 structure_generator_kwargs: Dict = None,

                 bandit_learner_class: Type[BanditLearner] = PseudoBandit,
                 bandit_learner_kwargs: Dict = None
                 ):
        """
        The Master class is responsible for the book keeping and to decide what to run next. Optimizers are
        instantiations of Master, that handle the important steps of deciding what configurations to run on what
        budget when.
        :param working_directory: The top level working directory accessible to all compute nodes(shared filesystem).
        :param logger: the logger to output some (more or less meaningful) information
        :param result_logger: a result logger that writes live results to disk
        """

        if bandit_learner_kwargs is None:
            bandit_learner_kwargs = {}
        if config_generator_kwargs is None:
            config_generator_kwargs = {}
        if structure_generator_kwargs is None:
            structure_generator_kwargs = {}

        self.working_directory = working_directory
        self.temp_dir = tempfile.TemporaryDirectory()
        if 'working_directory' not in config_generator_kwargs:
            config_generator_kwargs['working_directory'] = self.temp_dir.name

        if logger is None:
            self.logger = logging.getLogger('Master')
        else:
            self.logger = logger

        if result_logger is None:
            result_logger = ResultLogger(self.working_directory, self.temp_dir.name)
        self.result_logger = result_logger
        self.jobs = []

        # noinspection PyTypeChecker
        self.meta_information: MetaInformation = None

        self.ds = ds
        self.ds.cutoff = cutoff
        self.wallclock_limit = wallclock_limit
        self.cutoff = cutoff
        self.structure_cutoff_factor = structure_cutoff_factor
        self.pre_sample = pre_sample
        self.abort = False

        self.n_structures = 0

        # condition to synchronize the job_callback and the queue
        self.thread_cond = threading.Condition()
        self.incomplete_structures: Dict[CandidateId, Tuple[CandidateStructure, int, int]] = dict()

        if n_workers < 1:
            raise ValueError(f'Expected at least 1 worker, given {n_workers}')
        elif n_workers == 1 and cutoff <= 0:
            self.mgr: Optional[multiprocessing.Manager] = None
            self.cfg_cache: ConfigCache = ConfigCache(
                clazz=config_generator_class,
                init_kwargs=config_generator_kwargs,
                model=model)
            self.structure_generator: BaseStructureGenerator = structure_generator_class(
                cfg_cache=self.cfg_cache,
                cutoff=self.cutoff,
                workdir=self.working_directory,
                model=model,
                wallclock_limit=wallclock_limit,
                **structure_generator_kwargs)
        else:
            SyncManager.register('StructureGenerator', structure_generator_class)
            SyncManager.register('ConfigCache', ConfigCache)
            self.mgr: Optional[multiprocessing.Manager] = multiprocessing.Manager()
            # noinspection PyUnresolvedReferences
            self.cfg_cache: ConfigCache = self.mgr.ConfigCache(
                clazz=config_generator_class,
                init_kwargs=config_generator_kwargs,
                model=model)
            # noinspection PyUnresolvedReferences
            self.structure_generator: BaseStructureGenerator = self.mgr.StructureGenerator(
                cfg_cache=self.cfg_cache,
                cutoff=self.cutoff,
                workdir=self.working_directory,
                model=model,
                wallclock_limit=wallclock_limit,
                **structure_generator_kwargs)

        self.workers = []
        for i in range(n_workers):
            worker = worker_class(wid=str(i), cfg_cache=self.cfg_cache, workdir=self.temp_dir.name)
            self.workers.append(worker)

        self.dispatcher = Dispatcher(self.workers, self.structure_generator)
        self.bandit_learner: BanditLearner = bandit_learner_class(**bandit_learner_kwargs)

    def shutdown(self) -> None:
        self.logger.info('Shutdown initiated')
        # Sleep one second to guarantee dispatcher start, if startup procedure fails
        time.sleep(1)
        self.structure_generator.shutdown()
        self.dispatcher.shutdown()
        if self.mgr is not None:
            self.mgr.shutdown()

    def cleanup(self):
        self.temp_dir.cleanup()

    def optimize(self, fit: bool = True,
                 ensemble: bool = True,
                 render: bool = True,
                 store: bool = True) -> Union[Tuple[FlexiblePipeline, RunHistory],
                                              Tuple[FlexiblePipeline, RunHistory, PrefitVotingClassifier]]:
        """
        run optimization
        :return:
        """

        start = timeit.default_timer()
        start_time = datetime.datetime.now()

        data_file = os.path.join(self.working_directory, 'dataset.pkl')
        self.ds.store(data_file)

        self.meta_information = MetaInformation(start_time=time.time(), metric=self.ds.metric,
                                                openml_task=self.ds.task, openml_fold=self.ds.fold,
                                                data_file=os.path.abspath(data_file),
                                                config={
                                                    'cutoff': self.cutoff,
                                                    'wallclock_limit': self.wallclock_limit,
                                                })
        self.logger.info(f'starting run at {start_time:%Y-%m-%d %H:%M:%S}. Configuration:\n'
                         f'\twallclock_limit: {self.wallclock_limit}\n'
                         f'\tcutoff: {self.cutoff}\n'
                         f'\tpre_sample: {self.pre_sample}')
        for worker in self.workers:
            worker.start_time = start
        deadline = start + self.wallclock_limit

        def _optimize() -> bool:
            # Basic optimization logic without parallelism
            #   for candidate in self.bandit_learner.next_candidate():
            #       if candidate.is_proxy():
            #           candidate = self.structure_generator.fill_candidate(candidate, self.ds)
            #       n_configs = int(candidate.budget)
            #       for i in range(n_configs):
            #           if timeout:
            #               return True
            #           config = self.cfg_cache.sample_configuration([...])
            #           job = Job(candidate, config, [...])
            #           self.dispatcher.submit_job(job)
            #   return False

            it = self.bandit_learner.next_candidate()
            while True:
                if timeit.default_timer() > deadline:
                    self.logger.info("Timeout reached. Stopping optimization")
                    self.dispatcher.finish_work(self.cutoff)
                    return True
                if self.abort:
                    self.logger.info('Aborting optimization')
                    self.dispatcher.finish_work(self.cutoff)
                    return True
                if self.n_structures > 200:
                    return True

                job = None
                with self.thread_cond:
                    # Create EvaluationJob if possible
                    if len(self.incomplete_structures) > 0:
                        # TODO random selection mostly does not work as len(self.incomplete_structures) == 1
                        cid = random.choice(list(self.incomplete_structures.keys()))
                        candidate, n_configs, running = self.incomplete_structures[cid]

                        config_id = candidate.cid.with_config(len(candidate.results) + running)
                        if self.pre_sample:
                            config, cfg_key = self.cfg_cache.sample_configuration(
                                cid=config_id,
                                configspace=candidate.pipeline.configuration_space,
                                mf=self.ds.meta_features)
                            cfg_keys = [cfg_key]
                        else:
                            config = None
                            cfg_keys = candidate.cfg_keys

                        job = EvaluationJob(self.ds, config_id, candidate, self.cutoff, config, cfg_keys)
                        callback = self._evaluation_callback

                        if n_configs > 1:
                            self.incomplete_structures[cid] = candidate, n_configs - 1, running + 1
                        else:
                            del self.incomplete_structures[cid]
                    # Select new CandidateStructure if possible
                    else:
                        try:
                            candidate = next(it)
                            if candidate is None:
                                self.logger.debug(f'Waiting for next job to finish. '
                                                  f'Currently {len(self.dispatcher.running_jobs)} running, '
                                                  f'{self.bandit_learner.iterations[-1].num_running} outstanding')
                                # Safety-net to prevent infinite waiting
                                # noinspection PyTypeChecker
                                self.thread_cond.wait(max(self.cutoff, deadline - timeit.default_timer()))
                                continue

                            if candidate.is_proxy():
                                job = StructureJob(self.ds, candidate, self.structure_cutoff_factor * self.cutoff)
                                callback = self._structure_callback
                            else:
                                n_configs = int(candidate.budget)
                                self.incomplete_structures[candidate.cid] = candidate, n_configs, 0
                        except StopIteration:
                            # Current optimization is exhausted
                            return False

                if job is not None:
                    self.dispatcher.submit_job(job, callback)

        # while time_limit is not exhausted:
        #   structure, budget = structure_generator.get_next_structure()
        #   configspace = structure.configspace
        #
        #   incumbent, loss = bandit_learners.optimize(configspace, structure)
        #   Update score of selected structure with loss

        # Main hyperparameter optimization logic
        timeout = False
        repetition = 0
        offset = 0
        try:
            while not timeout:
                # noinspection PyTypeChecker
                self.dispatcher.finish_work(max(self.cutoff, deadline - timeit.default_timer()))
                self.logger.info(f'Starting repetition {repetition}')
                self.bandit_learner.reset(offset)
                timeout = _optimize()
                repetition += 1
                offset += len(self.bandit_learner.iterations)
        except KeyboardInterrupt:
            self.logger.info('Aborting optimization due to user interrupt')
        finally:
            structure_explanations = self.structure_generator.explain()
            config_explanations = self.cfg_cache.explain()
            self.shutdown()

        self.logger.info(f'Finished run after {(datetime.datetime.now() - start_time).seconds} seconds')

        iterations = self.result_logger.load()
        # noinspection PyAttributeOutsideInit
        self.rh_ = RunHistory.create(iterations, self.meta_information, self.bandit_learner.meta_data,
                                     os.path.join(self.working_directory, MODEL_DIR),
                                     structure_explanations, config_explanations)
        self.result_logger.log_run_history(self.rh_, str(self.meta_information.openml_task))

        pipeline, _ = self.rh_.get_incumbent()

        if fit:
            pipeline.fit(self.ds.X, self.ds.y)
        if render and fit:
            self.render(pipeline)
        if store and fit:
            joblib.dump(pipeline, os.path.join(self.working_directory, 'incumbent.pkl'))
        if ensemble:
            ensemble = self.build_ensemble(store=store)
            return pipeline, self.rh_, ensemble
        else:
            return pipeline, self.rh_

    def build_ensemble(self, ds: Dataset = None, store: bool = True) -> PrefitVotingClassifier:
        builder = EnsembleBuilder(self.working_directory, self.result_logger.structure_fn)
        ensemble = builder.fit(self.ds if ds is None else ds).get_ensemble()
        if store:
            self.result_logger.log_ensemble(ensemble, str(self.meta_information.openml_task))
        return ensemble

    def render(self, incumbent: FlexiblePipeline, ds: Dataset = None):
        renderer = NotebookRenderer()
        renderer.render(incumbent,
                        self.ds if ds is None else ds,
                        os.path.join(self.working_directory, 'incumbent.ipynb'))

    def _evaluation_callback(self, job: EvaluationJob) -> None:
        """
        method to be called when an evaluation has finished

        :param job: Finished Job
        :return:
        """
        self.logger.debug(f'Evaluation callback {job.cid}')
        with self.thread_cond:
            try:
                if job.config is None:
                    self.logger.error(
                        f'Encountered job without a configuration: {job.cid}. Using empty config as fallback')
                    config = ConfigurationSpace().get_default_configuration()
                    config.origin = 'Default'
                    job.config = config

                self.result_logger.log_evaluated_config(job.cs, job.result)
                cs = self.bandit_learner.register_result(job.cs, job.result)
                self.structure_generator.register_result(job.cs, job.result)
                self.cfg_cache.register_result(job)

                # Decrease number of running jobs
                if job.cs.cid in self.incomplete_structures:
                    _, n_configs, running = self.incomplete_structures[job.cs.cid]
                    self.incomplete_structures[job.cs.cid] = cs, n_configs, running - 1
            except KeyboardInterrupt:
                raise
            except (BrokenPipeError, EOFError) as ex:
                self.logger.fatal(f'Lost connection to SyncManager probably due to OOM. Aborting...: {ex}',
                                  exc_info=True)
                self.abort = True
            except Exception as ex:
                self.logger.fatal(f'Encountered unhandled exception {ex}. This should never happen!', exc_info=True)
            finally:
                self.thread_cond.notify_all()

    def _structure_callback(self, cs: CandidateStructure):
        self.logger.debug(f'Structure callback {cs.cid}')
        with self.thread_cond:
            try:
                if cs.is_proxy():
                    from dswizard.components.data_preprocessing.imputation import ImputationComponent
                    from dswizard.components.feature_preprocessing.one_hot_encoding import OneHotEncoderComponent
                    from dswizard.components.classification.decision_tree import DecisionTree
                    from dswizard.optimizers.structure_generators.fixed import FixedStructure

                    self.logger.warning('Encountered job without a structure. Using simple best-practice pipeline.')
                    cs = FixedStructure(steps=[('ohe', OneHotEncoderComponent()),
                                               ('imputation', ImputationComponent()),
                                               ('dt', DecisionTree())], cfg_cache=self.cfg_cache) \
                        .fill_candidate(cs, self.ds)

                self.result_logger.new_structure(cs)
                self.bandit_learner.iterations[-1].replace_proxy(cs)

                self.incomplete_structures[cs.cid] = cs, int(cs.budget), 0

                self.n_structures = self.n_structures + 1
            except KeyboardInterrupt:
                raise
            except (BrokenPipeError, EOFError) as ex:
                self.logger.fatal(f'Lost connection to SyncManager probably due to OOM. Aborting...: {ex}',
                                  exc_info=True)
                self.abort = True
            except Exception as ex:
                self.logger.fatal(f'Encountered unhandled exception {ex}. This should never happen!', exc_info=True)
            finally:
                self.thread_cond.notify_all()
