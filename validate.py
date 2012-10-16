from __future__ import division
import numpy as np

import joblib
import os
from delayed import delayed, run_delayed
from time import time
from crossvalidate import *


if __name__ == '__main__':
    import argparse
    import itertools
    from methods_list import make_methods_list

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=file, help='dataset to use')
    parser.add_argument('--n-jobs', dest='n_jobs', type=int,
                        help='number of total jobs to run', default=0)
    parser.add_argument('-j', type=int, help='job id to run', default=0)
    parser.add_argument('-n', type=int, help='number of datapoints to use', default=0)
    parser.add_argument('-a', help='flag to aggregate results', action='store_true')
    parser.add_argument('-f', help='flag to force recompute', action='store_true')
    args = parser.parse_args()

    # Load data
    X = np.load(args.dataset)
    X, y = X[:,:-1], X[:,-1]

    # Make list of methods
    methods = make_methods_list()
    M = len(methods)

    n = args.n if args.n else len(X)
    J = M
    n_jobs = min(J, args.n_jobs) if args.n_jobs else J

    if args.j < 0 or args.j >= n_jobs:
        raise ValueError('Job ID out of range.')

    # Range of jobs to run
    job_batchsize = int(J / n_jobs)
    a = args.j * job_batchsize
    b = a + job_batchsize if (args.j < n_jobs-1) else J

    train, test = range(n), range(n, len(X))

    if not args.a:
        # Setup list of jobs
        jobs = iter(delayed(run_method)(method, X, y, train, test, force=args.f)
                    for method in methods)

        # Run only jobs in batch
        for job in itertools.islice(jobs, a, b):
            run_delayed(job)

    else:      # Aggregate results
        validation = np.empty(M)
        wall = np.empty(M)
        validation.fill(np.nan)
        wall.fill(np.nan)

        # Fetch data
        jobs = iter(delayed(run_method)(method, X, y, train, test, load=True)
                    for method in methods)
        for i, job in enumerate(jobs):
            validation[i], wall[i] = run_delayed(job)

        # Save to `.npz` file
        datafname = args.dataset.name.split('/')[-1]
        fname = '{0:s}-k{1:02d}.validation'.format(
                    os.path.splitext(datafname)[0], args.k)
        np.savez(os.path.join(RESULTS, fname), validation=validation, wall=wall)
