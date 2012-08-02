"""
Make datasets that we're going to test on. This is just standardized so
that we just create a npy file in the same format np.c_[X, y]  for each
set.
"""

from __future__ import division
import numpy as np

import sklearn.datasets
import sklearn as sl

def make_dataset(name):
    if name == 'synthetic':
        return sklearn.datasets.make_classification(n_samples=10000, n_features=200,
                                                    n_informative=20, random_state=0)
    if name == 'digits':
        data = sklearn.datasets.load_digits()
        X, y = data.data, data.target
        i = np.random.permutation(X.shape[0])
        return X[i], y[i]

    else:
        raise RuntimeError('unknown dataset')

if __name__ == '__main__':
    import argparse
    import os, os.path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=str, help='dataset to create')
    args = parser.parse_args()

    np.random.seed(0)
    X, y = make_dataset(args.dataset)

    try:
        os.mkdir('data')
    except:
        pass

    np.save('data/' + args.dataset, np.c_[X, y])

