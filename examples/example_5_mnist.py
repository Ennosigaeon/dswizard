"""
Example 5 - MNIST
=================

Small CNN for MNIST implementet in both Keras and PyTorch.
This example also shows how to log results to disk during the optimization
which is useful for long runs, because intermediate results are directly available
for analysis. It also contains a more realistic search space with different types
of variables to be optimized.

"""
import argparse
import logging
import os
import pickle

import dswizard.core.nameserver as hpns
import dswizard.core.result as hpres
from dswizard.optimizers import BOHB

logging.basicConfig(level=logging.DEBUG)

parser = argparse.ArgumentParser(description='Example 5 - CNN on MNIST')
parser.add_argument('--min_budget', type=float, help='Minimum number of epochs for training.', default=1)
parser.add_argument('--max_budget', type=float, help='Maximum number of epochs for training.', default=9)
parser.add_argument('--worker', help='Flag to turn this into a worker process', action='store_true')
parser.add_argument('--run_id', type=str,
                    help='A unique run id for this optimization run. An easy option is to use the job id of the clusters scheduler.')
parser.add_argument('--nic_name', type=str, help='Which network interface to use for communication.', default='lo')
parser.add_argument('--shared_directory', type=str,
                    help='A directory that is accessible for all processes, e.g. a NFS share.', default='.')

args = parser.parse_args()

from example_5_keras_worker import KerasWorker as worker

# Every process has to lookup the hostname
host = hpns.nic_name_to_host(args.nic_name)

if args.worker:
    import time

    time.sleep(5)  # short artificial delay to make sure the nameserver is already running
    w = worker(run_id=args.run_id, host=host, timeout=120)
    w.load_nameserver_credentials(working_directory=args.shared_directory)
    w.run(background=False)
    exit(0)

# This example shows how to log live results. This is most useful
# for really long runs, where intermediate results could already be
# interesting. The core.result submodule contains the functionality to
# read the two generated files (results.json and configs.json) and
# create a Result object.
result_logger = hpres.JsonResultLogger(directory=args.shared_directory, overwrite=False)

# Start a nameserver:
NS = hpns.NameServer(run_id=args.run_id, host=host, port=0, working_directory=args.shared_directory)
ns_host, ns_port = NS.start()

# Start local worker
w = worker(run_id=args.run_id, host=host, nameserver=ns_host, nameserver_port=ns_port, timeout=120)
w.run(background=True)

# Run an optimizer
bohb = BOHB(configspace=worker.get_configspace(),
            run_id=args.run_id,
            host=host,
            nameserver=ns_host,
            nameserver_port=ns_port,
            result_logger=result_logger,
            min_budget=args.min_budget, max_budget=args.max_budget,
            )
res = bohb.run()

# store results
with open(os.path.join(args.shared_directory, 'results.pkl'), 'wb') as fh:
    pickle.dump(res, fh)

# shutdown
bohb.shutdown(shutdown_workers=True)
NS.shutdown()
