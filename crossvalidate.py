"""
Run leave-one-out cross-validation for some number of methods on the
given dataset and save the results.
"""

from __future__ import division
import numpy as np

import sklearn as sl
import sklearn.cross_validation

import itertools
from delayed import delayed, run_delayed
from time import time


def get_cv_indices(n, k, j):
    """
    Get indices for the `j`th fold of `k`-fold cross-validation on `n`
    data points.
    """
    cv = sl.cross_validation.LeaveOneOut(n) if (k == 0) else \
         sl.cross_validation.KFold(n, k)

    tr, ts = itertools.islice(iter(cv), j, j+1).next()

    return tr, ts


def run_method(methods, i, tr, ts, X, y):
    """
    Train the `i`th method in `methods` on `X[ts]` and test it on `X[tr]`.
    """
    clf = run_delayed(methods[i])

    start = time()
    clf.fit(X[tr], y[tr])
    accs = clf.score(X[ts], y[ts])
    wall = time() - start

    return accs, wall


import sklearn.svm
import sklearn.naive_bayes
import sklearn.ensemble
import sklearn.neighbors
import sklearn.tree
import sklearn.linear_model

methods = []

# SVMs.
methods.append(delayed(sl.svm.SVC)(kernel='rbf'))
methods.append(delayed(sl.svm.SVC)(kernel='linear'))
methods.append(delayed(sl.svm.SVC)(kernel='poly', degree=2))
methods.append(delayed(sl.svm.SVC)(kernel='poly', degree=3))

# Misc.
methods.append(delayed(sl.tree.DecisionTreeClassifier))
methods.append(delayed(sl.naive_bayes.GaussianNB))

# Random Forests.
for n in [5, 10, 15, 20, 25, 30]:
    methods.append(delayed(sl.ensemble.RandomForestClassifier)(n_estimators=n))

# Gradient boosting.
for a in [.0001, .001, .01, .1, .5]:
    methods.append(delayed(sl.ensemble.GradientBoostingClassifier)(learn_rate=a))

# Nearest neighbors.
for n in [5, 10, 15, 20, 25, 30]:
    methods.append(delayed(sl.neighbors.KNeighborsClassifier)(n_neighbors=n))

# l1-penalized logistic regression
for c in [10, 25, 50, 100, 250, 500, 1000]:
    methods.append(delayed(sl.linear_model.LogisticRegression)(C=c, penalty='l1', tol=0.01))

# l2-penalized logistic regression
for c in [10, 25, 50, 100, 250, 500, 1000]:
    methods.append(delayed(sl.linear_model.LogisticRegression)(C=c, penalty='l2', tol=0.01))


if __name__ == '__main__':
    import argparse
    import os.path
    import joblib
    import cPickle as pickle

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=file, help='dataset to use')
    parser.add_argument('-j', type=int, help='job id to run')
    parser.add_argument('-n', type=int, help='number of datapoints to use', default=0)
    parser.add_argument('-k', type=int, help='number of folds', default=0)

    args = parser.parse_args()

    # Get method and fold from job ID
    if args.k > 0:
        assert args.j < len(methods) * args.k, 'Job ID exceeds number of jobs.'
    m = int(args.j / len(methods))
    assert m < len(methods), 'Method ID exceeds number of methods.'
    fold = args.j % len(methods)

    # Load data
    X = np.load(args.dataset)
    y = X[:,-1]
    X = X[:,:-1]
    n = len(X) if (args.n == 0) else args.n

    tr, ts = get_cv_indices(n, args.k, fold)
    accs, wall = run_method(methods, m, tr, ts, X, y)

    # Save to file
    base = '%s-n%06d-k%03d-m%03d'.format(os.path.splitext(args.dataset.name)[0],
                                         n, args.k, m)
    fname = joblib.hashing.hash((args.dataset, methods, args.j))
    fname = os.path.join('results', base + fname + '.pkl')
    with open(fname, 'w-') as pklfile:
        pickle.dump((accs, wall), pklfile)

