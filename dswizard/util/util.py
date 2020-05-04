import importlib
from typing import Optional


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
