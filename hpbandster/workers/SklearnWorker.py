import importlib

from sklearn import metrics
from sklearn.model_selection import train_test_split

from components.pipeline import FlexiblePipeline
from core.model import ConfigId, ConfigInfo
from hpbandster.core import Worker


class SklearnWorker(Worker):

    def __init__(self, *args, sleep_interval=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.sleep_interval = sleep_interval

        self.dataset_properties = None
        self.X = None
        self.X_test = None
        self.y = None
        self.y_test = None

    def set_dataset(self, X, y, X_test=None, y_test=None, dataset_properties: dict = None, test_size: float = 0.3):
        if dataset_properties is None:
            dataset_properties = {}
        self.dataset_properties = dataset_properties

        if X_test is None:
            self.X, self.X_test, self.y, self.y_test = train_test_split(X, y, test_size=test_size)
        else:
            self.X = X
            self.X_test = X_test
            self.y = y
            self.y_test = y_test

    def compute(self, config_id: ConfigId, config: dict, config_info: ConfigInfo, budget: float,
                working_directory: str) -> dict:
        self.logger.info('Processing configuration {}'.format(config_id))
        # TODO budget missing

        pipeline = FlexiblePipeline(config_info.structure, self.dataset_properties)
        pipeline.set_hyperparameters(config)
        pipeline.fit(self.X, self.y)

        y_pred = pipeline.predict(self.X_test)
        res = metrics.accuracy_score(self.y_test, y_pred)

        return {
            'loss': res,
            'info': config_info
        }

    def create_estimator(self, conf: dict):
        try:
            name = conf['algorithm']
            kwargs = conf.copy()
            del kwargs['algorithm']

            module_name = name.rpartition(".")[0]
            class_name = name.split(".")[-1]

            module = importlib.import_module(module_name)
            class_ = getattr(module, class_name)
            return class_(**kwargs)
        except Exception as ex:
            self.logger.error('Invalid estimator with config {}'.format(conf))
            raise ex
