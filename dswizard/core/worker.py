from __future__ import annotations

import abc
import logging
import os
import socket
import threading
import traceback
from typing import Optional, TYPE_CHECKING

import Pyro4
import pynisher
from ConfigSpace import Configuration
from Pyro4.errors import CommunicationError, NamingError

from dswizard.core.config_cache import ConfigGeneratorCache
from dswizard.core.logger import ProcessLogger
from dswizard.core.model import Result, StatusType

if TYPE_CHECKING:
    from dswizard.components.pipeline import FlexiblePipeline
    from dswizard.core.base_config_generator import BaseConfigGenerator
    from dswizard.core.dispatcher import Dispatcher
    from dswizard.core.model import CandidateId


class Worker(abc.ABC):
    """
    The worker is responsible for evaluating a single configuration on a single budget at a time. Communication to the
    individual workers goes via the nameserver, management of the worker-pool and job scheduling is done by the
    Dispatcher and jobs are determined by the Master. In distributed systems, each cluster-node runs a Worker-instance.
    To implement your own worker, overwrite the `__init__`- and the `compute`-method. The first allows to perform
    initial computations, e.g. loading the dataset, when the worker is started, while the latter is repeatedly called
    during the optimization and evaluates a given configuration yielding the associated loss.
    """

    def __init__(self,
                 run_id: str,
                 nameserver: str = None,
                 nameserver_port: int = None,
                 host: str = None,
                 logger: logging.Logger = None,
                 wid: str = None,
                 workdir: str = '/tmp/dswizzard/'):
        """
        :param run_id: unique id to identify individual optimization run
        :param nameserver: hostname or IP of the nameserver
        :param nameserver_port: port of the nameserver
        :param logger: logger used for debugging output
        :param host: hostname for this worker process
        :param wid: if multiple workers are started in the same process, you MUST provide a unique id for each one of
            them using the `id` argument.
        """
        self.run_id = run_id
        self.host = host
        self.nameserver = nameserver
        self.nameserver_port = nameserver_port
        self.worker_id = '{}.worker.{}'.format(self.run_id, os.getpid())

        self.workdir = workdir
        self.process_logger = None

        if wid is not None:
            self.worker_id += '.{}'.format(wid)

        self.thread = None
        self.pyro_daemon = None

        if logger is None:
            self.logger = logging.getLogger(self.worker_id)
        else:
            self.logger = logger

        self.logger.info('Running on {} with pid {}'.format(socket.gethostname(), os.getpid()))

        self.busy = False
        self.thread_cond = threading.Condition(threading.Lock())

    def run(self, background: bool = False) -> None:
        """
        Method to start the worker.
        :param background: If set to False (Default). the worker is executed in the current thread. If True, a new
            daemon thread is created that runs the worker. This is useful in a single worker scenario/when the compute
            function only simulates work.
        :return:
        """
        if self.nameserver is None:
            return

        if background:
            self.worker_id += str(threading.get_ident())
            self.thread = threading.Thread(target=self._run, name='worker {} thread'.format(self.worker_id))
            self.thread.daemon = True
            self.thread.start()
        else:
            self._run()

    def _run(self):
        # initial ping to the dispatcher to register the worker

        try:
            with Pyro4.locateNS(host=self.nameserver, port=self.nameserver_port) as ns:
                self.logger.debug('Connected to nameserver {}'.format(ns))
                dispatchers = ns.list(prefix='{}.dispatcher'.format(self.run_id))
        except NamingError:
            if self.thread is None:
                raise RuntimeError('No nameserver found. Make sure the nameserver is running and '
                                   'that the host ({}) and port ({}) are correct'.format(self.nameserver,
                                                                                         self.nameserver_port))
            else:
                self.logger.error('No nameserver found. Make sure the nameserver is running and '
                                  'that the host ({}) and port ({}) are correct'.format(self.nameserver,
                                                                                        self.nameserver_port))
                exit(1)

        for dn, uri in dispatchers.items():
            try:
                self.logger.debug('found dispatcher {}'.format(dn))
                with Pyro4.Proxy(uri) as dispatcher_proxy:
                    dispatcher_proxy.trigger_discover_worker()

            except CommunicationError:
                self.logger.debug('Dispatcher did not respond. Waiting for one to initiate contact.')
                pass

        if len(dispatchers) == 0:
            self.logger.debug('No dispatcher found. Waiting for one to initiate contact.')

        self.logger.info('start listening for jobs')

        self.pyro_daemon = Pyro4.core.Daemon(host=self.host)

        with Pyro4.locateNS(self.nameserver, port=self.nameserver_port) as ns:
            uri = self.pyro_daemon.register(self, self.worker_id)
            ns.register(self.worker_id, uri)

        self.pyro_daemon.requestLoop()

        with Pyro4.locateNS(self.nameserver, port=self.nameserver_port) as ns:
            ns.remove(self.worker_id)

    @Pyro4.expose
    @Pyro4.oneway
    def start_computation(self,
                          callback: Dispatcher,
                          cid: CandidateId,
                          config: Optional[Configuration],
                          pipeline: FlexiblePipeline,
                          budget: float,
                          timeout: float = None) -> Result:
        with self.thread_cond:
            while self.busy:
                self.thread_cond.wait()
            self.busy = True

        self.logger.info('start processing job {}'.format(cid))

        result = None
        try:
            # On the fly configuration of pipeline
            if config is None:
                cfg = self._get_config_generator(cid, pipeline)
            else:
                cfg = None

            wrapper = pynisher.enforce_limits(wall_time_in_s=timeout)(self.compute)
            c = wrapper(cid, config, cfg, pipeline, budget)

            if wrapper.exit_status is pynisher.TimeoutException:
                status = StatusType.TIMEOUT
                cost = 1
            elif wrapper.exit_status is pynisher.MemorylimitException:
                status = StatusType.MEMOUT
                cost = 1
            elif wrapper.exit_status == 0 and c is not None:
                status = StatusType.SUCCESS
                cost = c
            else:
                status = StatusType.CRASHED
                cost = 1

            if config is None:
                config, partial_configs = self.process_logger.restore_config(pipeline)
            else:
                partial_configs = None

            runtime = float(wrapper.wall_clock_time)
            result = Result(status, config, cost, runtime, partial_configs)
        except KeyboardInterrupt:
            raise
        except Exception:
            # Should never occur, just a safety net
            self.logger.error('Unexpected error during computation: \'{}\''.format(traceback.format_exc()))
            result = Result(StatusType.CRASHED, config, 1, None, None)
        finally:
            self.process_logger = None
            self.logger.debug('done with job {}, trying to register results with dispatcher.'.format(cid))
            with self.thread_cond:
                self.busy = False
                callback.register_result(cid, result)
                self.thread_cond.notify()
        return result

    def _get_config_generator(self, cid: CandidateId, pipeline: FlexiblePipeline) -> Optional[BaseConfigGenerator]:
        cache: Optional[ConfigGeneratorCache] = None
        if self.nameserver is None:
            cache: ConfigGeneratorCache = ConfigGeneratorCache.instance()
        else:
            with Pyro4.locateNS(host=self.nameserver, port=self.nameserver_port) as ns:
                uri = list(ns.list(prefix='{}.config_generator'.format(self.run_id)).values())
                if len(uri) != 1:
                    raise ValueError('Expected exactly one ConfigGeneratorCache but found {}'.format(len(uri)))
                # noinspection PyTypeChecker
                cache = Pyro4.Proxy(uri[0])

        cfg = cache.get(pipeline.configuration_space, pipeline=pipeline)
        self.process_logger = ProcessLogger(self.workdir, cid)
        return cfg

    @abc.abstractmethod
    def compute(self,
                config_id: CandidateId,
                config: Optional[Configuration],
                cfg: Optional[BaseConfigGenerator],
                pipeline: FlexiblePipeline,
                budget: float
                ) -> float:
        """
        The function you have to overload implementing your computation.
        :param config_id: the id of the configuration to be evaluated
        :param config: the actual configuration to be evaluated.
        :param cfg:
        :param pipeline: Additional information about the sampled configuration like pipeline structure.
        :param budget: the budget for the evaluation
        """
        pass

    @Pyro4.expose
    def is_busy(self) -> bool:
        return self.busy

    @Pyro4.expose
    @Pyro4.oneway
    def shutdown(self) -> None:
        self.logger.debug('shutting down')

        if self.pyro_daemon is not None:
            self.pyro_daemon.shutdown()
        if self.thread is not None:
            self.thread.join()
