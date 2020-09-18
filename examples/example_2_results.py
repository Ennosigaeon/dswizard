from core.logger import JsonResultLogger
from core.runhistory import RunHistory

logger = JsonResultLogger('run/logs/', init=False)

rh = RunHistory(logger.load(), {})
pipeline, cs = rh.get_incumbent()
