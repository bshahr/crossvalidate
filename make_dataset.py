"""
Make datasets that we're going to test on. This is just standardized so
that we just create a npy file in the same format np.c_[X, y]  for each
set.
"""

from __future__ import division
import numpy as np
import os

import sklearn.datasets
import sklearn as sl


DATA = os.environ.get('PYCROSSVALIDATE_DATA', 'data')

def make_dataset(name):
    if name == 'synthetic':
        return sklearn.datasets.make_classification(n_samples=10000,
                                                    n_features=200,
                                                    n_informative=20,
                                                    random_state=0)

    if name == 'spambase':
        data = np.loadtxt(os.path.join(DATA, 'spambase.data'), delimiter=',')
        np.random.shuffle(data)
        X, y = data[:,:-1], data[:,-1]
        return X, y

    if name == 'digits':
        data = sklearn.datasets.load_digits()
        X, y = data.data, data.target
        i = np.random.permutation(X.shape[0])
        return X[i], y[i]

    if name == 'cover-binary':
        data = np.loadtxt(os.path.join(DATA, 'covtype.data'), delimiter=',')
        np.random.shuffle(data)
        idx = np.nonzero((data[:,-1] == 1) + (data[:,-1] == 2))[0]
        X, y = data[idx,:-1], data[idx,-1]
        return X, y

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
        os.mkdir(DATA)
    except:
        pass

    np.save(os.path.join(DATA, args.dataset), np.c_[X, y])

