import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import Bunch
from sklearn.utils.multiclass import check_classification_targets


class PrefitVotingClassifier(VotingClassifier):

    # noinspection PyAttributeOutsideInit
    def fit(self, X, y, sample_weight=None) -> 'PrefitVotingClassifier':
        """Fit the estimators.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like of shape (n_samples,)
            Target values.

        sample_weight : array-like of shape (n_samples,), default=None
            Sample weights. If None, then samples are equally weighted.
            Note that this is supported only if all underlying estimators
            support sample weights.

            .. versionadded:: 0.18

        Returns
        -------
        self : object

        """
        check_classification_targets(y)
        if isinstance(y, np.ndarray) and len(y.shape) > 1 and y.shape[1] > 1:
            raise NotImplementedError('Multilabel and multi-output'
                                      ' classification is not supported.')

        if self.voting not in ('soft', 'hard'):
            raise ValueError("Voting must be 'soft' or 'hard'; got (voting=%r)"
                             % self.voting)

        self.le_ = LabelEncoder().fit(y)
        self.classes_ = self.le_.classes_

        """Get common fit operations."""
        names, clfs = self._validate_estimators()

        if (self.weights is not None and
                len(self.weights) != len(self.estimators)):
            raise ValueError('Number of `estimators` and weights must be equal'
                             '; got %d weights, %d estimators'
                             % (len(self.weights), len(self.estimators)))

        self.estimators_ = [clf for idx, clf in enumerate(clfs) if clf not in (None, 'drop')]
        self.named_estimators_ = Bunch()

        # Uses None or 'drop' as placeholder for dropped estimators
        est_iter = iter(self.estimators_)
        for name, est in self.estimators:
            current_est = est if est in (None, 'drop') else next(est_iter)
            self.named_estimators_[name] = current_est

        return self
