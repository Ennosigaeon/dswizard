from typing import Optional

from dswizard.core.config_cache import ConfigCache

_cfg_cache_instance: ConfigCache = None


# TODO only adhoc. needs to be replaced
def get_config_generator_cache() -> Optional[ConfigCache]:
    return _cfg_cache_instance
