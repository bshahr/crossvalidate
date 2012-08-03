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
from methods_list import make_methods_list


#RESULTSPATH = os.environ['PYCROSSVALIDATE_RESULTSPATH'] or 'results'
RESULTSPATH = 'results'

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
    methods = make_methods_list()
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
    test = accs.mean(axis=1)
    cv = sl.cross_validation.LeaveOneOut(n) if (args.k == 0) else \
         sl.cross_validation.KFold(n, args.k)
    fold_size = np.array([len(fold) for _, fold in cv])
    success = accs * fold_size
    failure = fold_size - success

    np.savez(base, success=success, failure=failure, test=test, wall=wall)
