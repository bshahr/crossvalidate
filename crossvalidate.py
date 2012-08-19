"""
Run leave-one-out cross-validation for some number of methods on the
given dataset and save the results.
"""

from __future__ import division
import numpy as np

import joblib
import os
from delayed import delayed, run_delayed
from time import time


RESULTS = os.environ.get('PYCROSSVALIDATE_RESULTS', 'results')
CACHE = os.environ.get('PYCROSSVALIDATE_CACHE', 'cache')

memory = joblib.Memory(cachedir=CACHE, verbose=0)
@memory.cache
def _run_method(method, X, y, train, test):
    """
    Train the `method` on X[train] and test on X[test].
    """
    clf = run_delayed(method)

    start = time()
    clf.fit(X[train], y[train])
    accs = clf.score(X[test], y[test])
    wall = time() - start

    return accs, wall

def run_method(method, X, y, train, test, force=False, load=False):
    """
    Wrapper around the MemorizedFunc. This allows run_method to be delayed.
    Parameters:
        method, X, y, train, test - arguments passed to _run_method;
        force - boolean to force a rerun with the current arguments;
        load - boolean to prevent a rerun and only allow loading,
               ignored if force=True.
    Returns:
        accs - accuracy of method on fold `X[test]`;
        wall - wall time spent in computation.
    """
    params = (method, X, y, train, test)

    if force:
        return _run_method.call(*params)
    elif load:
        outdir = _run_method.get_output_dir(*params)[0]
        try:
            return _run_method.load_output(outdir)
        except IOError:
            return (np.nan, np.nan)
    else:
        return _run_method(*params)


if __name__ == '__main__':
    import argparse
    import itertools
    from sklearn.cross_validation import KFold, LeaveOneOut
    from methods_list import make_methods_list

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=file, help='dataset to use')
    parser.add_argument('--n-jobs', dest='n_jobs', type=int,
                        help='number of total jobs to run', default=0)
    parser.add_argument('-j', type=int, help='job id to run', default=0)
    parser.add_argument('-n', type=int, help='number of datapoints to use', default=0)
    parser.add_argument('-k', type=int, help='number of folds', default=0)
    parser.add_argument('-a', help='flag to aggregate results', action='store_true')
    parser.add_argument('-f', help='flag to force recompute', action='store_true')
    args = parser.parse_args()

    # Load data
    X = np.load(args.dataset)
    X, y = X[:,:-1], X[:,-1]

    # Make list of methods
    methods = make_methods_list()

    n = args.n if args.n else len(X)
    M = len(methods)
    J = args.k * M if args.k else n * M
    n_jobs = min(J, args.n_jobs) if args.n_jobs else J

    if args.j < 0 or args.j >= n_jobs:
        raise ValueError('Job ID out of range.')

    # Range of jobs to run
    job_batchsize = int(J / n_jobs)
    a = args.j * job_batchsize
    b = a + job_batchsize if (args.j < n_jobs-1) else J

    # Get cross-validation folds
    cv = LeaveOneOut(n) if (args.k == 0) else KFold(n, args.k)


    if not args.a:
        # Setup list of jobs
        jobs = iter(delayed(run_method)(method, X, y, train, test, force=args.f)
            	    for method, (train, test) in itertools.product(methods, cv))
        
        # Run only jobs in batch
        for job in itertools.islice(jobs, a, b):
            run_delayed(job)

    else:      # Aggregate results
        accs = np.empty(M * args.k)
        wall = np.empty(M * args.k)
        accs.fill(np.nan)
        wall.fill(np.nan)

        # Fetch data
        jobs = iter(delayed(run_method)(method, X, y, train, test, load=True)
                    for method, (train, test) in itertools.product(methods, cv))
        for i, job in enumerate(jobs):
            accs[i], wall[i] = run_delayed(job)

        accs.resize((M, args.k))
        wall.resize((M, args.k))
        test = accs.mean(axis=1)

        fold_size = np.array([len(fold) for _, fold in cv])
        success = accs * fold_size
        failure = fold_size - success

        # Save to `.npz` file
        datafname = args.dataset.name.split('/')[-1]
        fname = '{0:s}-k{1:02d}'.format(
                    os.path.splitext(datafname)[0],
                    args.k)
        np.savez(os.path.join(RESULTS, fname),
                 success=success, failure=failure, test=test, wall=wall)
