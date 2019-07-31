import logging
import multiprocessing
import os
import pickle
import socket
import threading
import time
import traceback

import Pyro4
from Pyro4.errors import CommunicationError, NamingError

from hpbandster.core.model import ConfigId, ConfigInfo, Result
from util.process import Process


class Worker(object):
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
                 logger: logging.Logger = None,
                 host: str = None,
                 id: any = None,
                 timeout: float = None):
        """
        :param run_id: unique id to identify individual HpBandSter run
        :param nameserver: hostname or IP of the nameserver
        :param nameserver_port: port of the nameserver
        :param logger: logger used for debugging output
        :param host: hostname for this worker process
        :param id: if multiple workers are started in the same process, you MUST provide a unique id for each one of
            them using the `id` argument.
        :param timeout: specifies the timeout a worker will wait for a new after finishing a computation before shutting
            down. Towards the end of a long run with multiple workers, this helps to shutdown idling workers. We
            recommend a timeout that is roughly half the time it would take for the second largest budget to finish.
            The default (None) means that the worker will wait indefinitely and never shutdown on its own.
        """
        self.run_id = run_id
        self.host = host
        self.nameserver = nameserver
        self.nameserver_port = nameserver_port
        self.worker_id = 'hpbandster.run_{}.worker.{}.{}'.format(self.run_id, socket.gethostname(), os.getpid())

        self.timeout = timeout
        self.timer = None

        if id is not None:
            self.worker_id += '.{}'.format(id)

        self.thread = None

        if logger is None:
            self.logger = logging.getLogger(self.worker_id)
        else:
            self.logger = logger

        self.logger.info('Running on {} with pid {}'.format(socket.gethostname(), os.getpid()))

        self.busy = False
        self.thread_cond = threading.Condition(threading.Lock())
        self.manager = multiprocessing.Manager()

    def load_nameserver_credentials(self,
                                    working_directory: str,
                                    num_tries: int = 60,
                                    interval: int = 1) -> None:
        """
        loads the nameserver credentials in cases where master and workers share a filesystem
        :param working_directory: the working directory for the HPB run (see master)
        :param num_tries: number of attempts to find the file (default 60)
        :param interval: waiting period between the attempts
        :return:
        """
        fn = os.path.join(working_directory, 'HPB_run_{}_pyro.pkl'.format(self.run_id))

        for i in range(num_tries):
            try:
                with open(fn, 'rb') as fh:
                    self.nameserver, self.nameserver_port = pickle.load(fh)
                return
            except FileNotFoundError:
                self.logger.warning('config file {} not found (trail {}/{})'.format(fn, i + 1, num_tries))
                time.sleep(interval)
        raise RuntimeError('Could not find the nameserver information, aborting!')

    def run(self, background: bool = False) -> None:
        """
        Method to start the worker.
        :param background: If set to False (Default). the worker is executed in the current thread. If True, a new
            daemon thread is created that runs the worker. This is useful in a single worker scenario/when the compute
            function only simulates work.
        :return:
        """
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
                dispatchers = ns.list(prefix='hpbandster.run_{}.dispatcher'.format(self.run_id))
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

    def compute(self,
                config_id: ConfigId,
                config: dict,
                config_info: ConfigInfo,
                budget: float,
                working_directory: str,
                result: dict) -> None:
        """
        The function you have to overload implementing your computation.
        :param config_id: the id of the configuration to be evaluated
        :param config: the actual configuration to be evaluated.
        :param config_info: Additional information about the sampled configuration like pipeline structure.
        :param budget: the budget for the evaluation
        :param working_directory: a name of a directory that is unique to this configuration. Use this to store
            intermediate results on lower  budgets that can be reused later for a larger budget (for iterative
            algorithms, for example).
        :param result: contains the return values in a dictionary. Needs to contain two mandatory entries:
                - 'loss': a numerical value that is MINIMIZED
                - 'info': This can be pretty much any build in python type, e.g. a dict with lists as value. Due to
                          Pyro4 handling the remote function calls, 3rd party types like numpy arrays are not supported!
        """

        raise NotImplementedError(
            'Subclass hpbandster.distributed.worker and overwrite the compute method in your worker script')

    @Pyro4.expose
    @Pyro4.oneway
    def start_computation(self,
                          callback,
                          id: ConfigId,
                          timeout: float = None,
                          *args,
                          **kwargs) -> Result:
        with self.thread_cond:
            while self.busy:
                self.thread_cond.wait()
            self.busy = True
        if self.timeout is not None and self.timer is not None:
            self.timer.cancel()

        self.logger.info('start processing job {}'.format(id))
        self.logger.debug('args: {}'.format(args))
        self.logger.debug('kwargs: {}'.format(kwargs))
        result = None
        try:


            p = Process(target=self.compute, args=args, kwargs={'config_id': id, 'result': d, **kwargs})
            p.start()
            p.join(timeout)

            if p.is_alive():
                self.logger.debug('Abort fitting after timeout')
                p.terminate()
                p.join()
                result = Result.failure('Computation did not finish within {} seconds'.format(timeout))

            if p.exception:
                error, tb = p.exception
                result = Result.failure(tb)
            else:
                # noinspection PyProtectedMember
                result = Result.success(d._getvalue())

        except Exception:
            # Should never occur, just a safety net
            self.logger.error('Unexpected error during computation: \'{}\''.format(traceback.format_exc()))
            result = Result.failure(
                traceback.format_exc()
            )
        finally:
            self.logger.debug('done with job {}, trying to register it.'.format(id))
            with self.thread_cond:
                self.busy = False
                callback.register_result(id, result)
                self.thread_cond.notify()
        self.logger.info('registered result for job {} with dispatcher'.format(id))
        if self.timeout is not None:
            self.timer = threading.Timer(self.timeout, self.shutdown)
            self.timer.daemon = True
            self.timer.start()
        return result

    @Pyro4.expose
    def is_busy(self) -> bool:
        return self.busy

    @Pyro4.expose
    @Pyro4.oneway
    def shutdown(self) -> None:
        self.logger.debug('shutting down now!')
        self.pyro_daemon.shutdown()
        if self.thread is not None:
            self.thread.join()
