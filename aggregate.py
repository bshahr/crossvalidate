"""
Run leave-one-out cross-validation for some number of methods on the
given dataset and save the results.
"""

from __future__ import division
import numpy as np

import sklearn as sl

import itertools
import joblib
import os
from delayed import delayed, run_delayed
from time import time


#RESULTSPATH = os.environ['PYCROSSVALIDATE_RESULTSPATH'] or 'results'
RESULTSPATH = 'results'

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
methods.append(delayed(sl.tree.DecisionTreeClassifier)())
methods.append(delayed(sl.naive_bayes.GaussianNB)())

# Random Forests.
for n in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200]:
    methods.append(delayed(sl.ensemble.RandomForestClassifier)(n_estimators=n))

# Gradient boosting.
for a in [.0001, .001, .01, .1, .5, 1., 5., 10.]:
    methods.append(delayed(sl.ensemble.GradientBoostingClassifier)(learn_rate=a))

# Nearest neighbors.
for n in [1, 2, 5, 10, 15, 20, 25, 30, 50]:
    methods.append(delayed(sl.neighbors.KNeighborsClassifier)(n_neighbors=n))

# l1-penalized logistic regression
for c in [0.1, 0.5, 1, 5, 10, 25, 50, 100, 250, 500, 1000, 2000, 4000]:
    methods.append(delayed(sl.linear_model.LogisticRegression)(C=c, penalty='l1', tol=0.01))

# l2-penalized logistic regression
for c in [0.1, 0.5, 1, 5, 10, 25, 50, 100, 250, 500, 1000, 2000, 4000]:
    methods.append(delayed(sl.linear_model.LogisticRegression)(C=c, penalty='l2', tol=0.01))


if __name__ == '__main__':
    import argparse
    import os
    import cPickle as pickle
    import itertools
    import sklearn.cross_validation

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=file, help='dataset to use')
    parser.add_argument('-n', type=int, help='number of datapoints to use', default=0)
    parser.add_argument('-k', type=int, help='number of folds', default=0)
    args = parser.parse_args()

    # Load data
    X = np.load(args.dataset)
    n = args.n if args.n else len(X)
    M = len(methods)

    # Create results directory
    datafname = args.dataset.name.split('/')[-1]        # remove data directory
    base = '{0:s}-k{1:02d}'.format(
                os.path.splitext(datafname)[0],
                args.k
                )
    directory = os.path.join(RESULTSPATH, base)
    try:
        os.chdir(directory)
    except OSError:
        print 'Cannot find dataset {0:s} with k={1:02d}.'.format(args.dataset,
                                                                 args.k)

    # Get cross-validation folds
    cv = sl.cross_validation.LeaveOneOut(n) if (args.k == 0) else \
         sl.cross_validation.KFold(n, args.k)

    # Setup list of jobs
    files = iter(joblib.hashing.hash((args.dataset, method, n, args.k, train, test))
                 + '.pkl'
                 for method, (train, test) in itertools.product(methods, cv))

    # Open pickles and assign to matrices
    accs = np.empty(M * args.k)
    wall = np.empty(M * args.k)
    accs.fill(np.nan)       # if the file does not exist set to np.nan
    wall.fill(np.nan)
    for i, file_ in enumerate(files):
        try:
            with open(file_, 'rb') as pklfile:
                accs[i], wall[i] = pickle.load(pklfile)
        except IOError:
            pass

    accs.resize((M, args.k))
    wall.resize((M, args.k))
    cv = sl.cross_validation.LeaveOneOut(n) if (args.k == 0) else \
         sl.cross_validation.KFold(n, args.k)
    fold_size = [len(fold) for _, fold in cv]

    np.savez(base, accs=accs, wall=wall, fold_size=fold_size)
