import threading
from typing import Optional

import Pyro4
import Pyro4.naming

from dswizard.core.config_cache import ConfigCache


def nic_name_to_host(nic_name):
    """ translates the name of a network card into a valid host name"""
    from netifaces import ifaddresses, AF_INET
    host = ifaddresses(nic_name).setdefault(AF_INET, [{'addr': 'No IP addr'}])[0]['addr']
    return host


def start_local_nameserver(host=None, port=0, nic_name=None):
    """
    starts a Pyro4 nameserver in a daemon thread
    :param host: the hostname to use for the nameserver
    :param port: the port to be used. Default =0 means a random port
    :param nic_name: name of the network interface to use
    :return: the host name and the used port
    """

    if host is None:
        if nic_name is None:
            host = 'localhost'
        else:
            host = nic_name_to_host(nic_name)

    uri, ns, _ = Pyro4.naming.startNS(host=host, port=port)
    host, port = ns.locationStr.split(':')

    thread = threading.Thread(target=ns.requestLoop, name='Pyro4 nameserver started by dswizard')
    thread.daemon = True

    thread.start()
    return host, int(port)


_cfg_cache_instance: ConfigCache = None


def get_config_generator_cache(nameserver: str, port: int, run_id: str) -> Optional[ConfigCache]:
    if nameserver is None:
        return _cfg_cache_instance
    else:
        with Pyro4.locateNS(host=nameserver, port=port) as ns:
            uri = list(ns.list(prefix='{}.config_generator'.format(run_id)).values())
            if len(uri) != 1:
                raise ValueError('Expected exactly one ConfigCache but found {}'.format(len(uri)))
            # noinspection PyTypeChecker
            return Pyro4.Proxy(uri[0])
