"""
Run leave-one-out cross-validation for some number of methods on the
given dataset and save the results.
"""

from __future__ import division
import numpy as np

import itertools
import joblib
import os
from delayed import delayed, run_delayed
from time import time


#RESULTSPATH = os.environ['PYCROSSVALIDATE_RESULTSPATH'] or 'results'
RESULTSPATH = 'results'

def run_method(method, train, test, X, y, directory, pass_to_hash):
    """
    Train the `method` on X[train] and test on X[test].
    `pass_to_hash` is a tuple that is passed to joblib's
    hasing function.
    """
    # Check whether this had already been run
    fname = joblib.hashing.hash(pass_to_hash)
    fname = os.path.join(directory, fname + '.pkl')
    try:
        with open(fname, 'rb') as pklfile:
            return None
    except IOError:
        pass

    clf = run_delayed(method)

    start = time()
    clf.fit(X[train], y[train])
    accs = clf.score(X[test], y[test])
    wall = time() - start

    with open(fname, 'wb-') as pklfile:
        pickle.dump((accs, wall), pklfile)


if __name__ == '__main__':
    import argparse
    import os
    import cPickle as pickle
    import itertools
    import sklearn.cross_validation
    from methods_list import make_methods_list

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=file, help='dataset to use')
    parser.add_argument('--n-jobs', dest='n_jobs', type=int,
                        help='number of total jobs to run', default=0)
    parser.add_argument('-j', type=int, help='job id to run')
    parser.add_argument('-n', type=int, help='number of datapoints to use', default=0)
    parser.add_argument('-k', type=int, help='number of folds', default=0)
    args = parser.parse_args()

    # Load data
    X = np.load(args.dataset)
    y = X[:,-1]
    X = X[:,:-1]

    # Make list of methods
    methods = make_methods_list()

    n = args.n if args.n else len(X)
    M = len(methods)
    J = args.k * M if args.k else n * M
    n_jobs = min(J, args.n_jobs) if args.n_jobs else J

    if args.j < 0 or args.j >= n_jobs:
        raise ValueError('Job ID out of range.')

    # Create results directory
    datafname = args.dataset.name.split('/')[-1]        # remove data directory
    base = '{0:s}-k{1:02d}'.format(
                os.path.splitext(datafname)[0],
                args.k
                )
    directory = os.path.join(RESULTSPATH, base)
    try:
        os.mkdir(directory)
    except OSError:
        pass

    # Range of jobs to run
    job_batchsize = int(J / n_jobs)
    a = args.j * job_batchsize
    b = a + job_batchsize if (args.j < n_jobs-1) else J

    # Get cross-validation folds
    cv = sklearn.cross_validation.LeaveOneOut(n) if (args.k == 0) else \
         sklearn.cross_validation.KFold(n, args.k)

    # Setup list of jobs
    jobs = iter(delayed(run_method)(method, train, test, X, y,
                                    directory=directory,
                                    pass_to_hash=(args.dataset, method, n,
                                                  args.k, train, test))
                for method, (train, test) in itertools.product(methods, cv))

    # Run only jobs in batch
    for job in itertools.islice(jobs, a, b):
        run_delayed(job)


