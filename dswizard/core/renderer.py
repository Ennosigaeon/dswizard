import os
import textwrap
from collections import defaultdict

import joblib
import nbformat.v4
from nbformat import NotebookNode
from sklearn.base import BaseEstimator
from typing import List

from dswizard.core.model import Dataset
from dswizard.pipeline.pipeline import FlexiblePipeline


class NotebookRenderer:
    class PipelineStep:

        def __init__(self, name: str, estimator: BaseEstimator):
            self.name = name
            self.estimator = estimator
            self.module = estimator.__class__.__module__
            self.qualname = estimator.__class__.__qualname__

    def __init__(self):
        self.reset()
        self.WHITESPACE = '                                '

    # noinspection PyAttributeOutsideInit
    def reset(self):
        self.indent: int = 0
        self.sb: List[str] = []

    def render(self, pipeline: FlexiblePipeline, ds: Dataset, fname: str):
        self.reset()

        steps = []
        for name, step in pipeline.steps:
            if step.estimator_ is None:
                raise ValueError(f'Step {name} is not fitted yet. Unable to render pipeline')
            steps.append(NotebookRenderer.PipelineStep(name, step.estimator_))

        notebook = nbformat.v4.new_notebook()
        cells = [
            self._imports(steps),
            self._dataset(ds),
            self._pipeline(steps),
            self._predictions(ds.metric),
        ]
        notebook['cells'] = cells

        with open(fname, 'w') as f:
            nbformat.write(notebook, f)

    def _flush(self) -> str:
        code = '\n'.join(self.sb)
        self.reset()
        return code

    def _imports(self, steps: List[PipelineStep]) -> NotebookNode:
        """
        from sklearn.pipeline import Pipeline
        """
        self._write('import openml')
        modules = defaultdict(list)
        modules['openml'].append('OpenMLSupervisedTask')
        modules['sklearn.pipeline'].append('Pipeline')
        modules['sklearn'].append('metrics')
        modules['sklearn.utils.multiclass'].append('type_of_target')

        for step in steps:
            modules[str(step.module).split('._')[0]].append(step.qualname)
        for module, items in modules.items():
            self._write(f'from {module} import {", ".join(items)}')
        self._linebreak()

        return nbformat.v4.new_code_cell(self._flush())

    def _dataset(self, ds: Dataset):
        self._write(f'''
        # Load data from OpenML and create train/test splits
        # noinspection PyTypeChecker
        task: OpenMLSupervisedTask = openml.tasks.get_task({ds.task})
        train_indices, test_indices = task.get_train_test_split_indices(fold={ds.fold})

        X, y = task.get_X_and_y()
        X_train = X[train_indices, :]
        y_train = y[train_indices]
        X_test = X[test_indices, :]
        y_test = y[test_indices]
        ''')

        return nbformat.v4.new_code_cell(self._flush())

    def _pipeline(self, steps: List[PipelineStep]):
        self._write('# Construct ML pipeline')
        self._write(f'pipeline = Pipeline([')
        with self:
            for step in steps:
                self._write(f"('{step.name}', {step.estimator}),")
        self._write('])')
        self._write('pipeline = pipeline.fit(X_train, y_train)')

        return nbformat.v4.new_code_cell(self._flush())

    def _predictions(self, metric: str):
        self._write('''
        # Create predictions of test data
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)
        ''')

        if metric == 'accuracy':
            self._write('score = metrics.accuracy_score(y_test, y_pred)')
        elif metric == 'precision':
            self._write('metrics.precision_score(y_test, y_pred, average="weighted")')
        elif metric == 'recall':
            self._write('metrics.recall_score(y_test, y_pred, average="weighted")')
        elif metric == 'f1':
            self._write('score = metrics.f1_score(y_test, y_pred, average="weighted")')
        elif metric == 'logloss':
            self._write('score = metrics.log_loss(y_test, y_prob)')
        elif metric == 'roc_auc':
            self._write('''
            y_type = type_of_target(y_test)
            if y_type == "binary" and y_prob.ndim > 1:
                y_prob_ = y_prob[:, 1]
            else:
                y_prob_ = y_prob
            score = metrics.roc_auc_score(y_test, y_prob_, average='weighted', multi_class='ovr')
            ''')
        self._write("print('Performance', score)")

        return nbformat.v4.new_code_cell(self._flush())

    def _linebreak(self, count: int = 1):
        for i in range(count):
            self._write('')

    def _write(self, line: str):
        trimmed = textwrap.dedent(line)
        self.sb.append(textwrap.indent(trimmed, self.WHITESPACE[:self.indent]))

    def __enter__(self):
        self.indent += 4

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.indent -= 4
        assert self.indent >= 0


def main():
    workdir = '../../scripts/run/59/'
    pipeline = joblib.load(os.path.join(workdir, 'incumbent.pkl'))
    ds = joblib.load(os.path.join(workdir, 'dataset.pkl'))

    renderer = NotebookRenderer()
    renderer.render(pipeline, ds, os.path.join(workdir, 'test.ipynb'))


if __name__ == '__main__':
    main()
