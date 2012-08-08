"""
Create list of delayed classifiers to be used by `crossvalidate.py` and
`aggregate.py`.
"""
from __future__ import division

import sklearn as sl
import sklearn.svm
import sklearn.naive_bayes
import sklearn.ensemble
import sklearn.neighbors
import sklearn.tree
import sklearn.linear_model

from delayed import delayed

def make_methods_list():
    methods = []
    """
    # SVMs.
    for C in [1e2, 1e3, 1e4]:
        for gamma in [0.01, 0.05, 0.1]:
            methods.append(delayed(sl.svm.SVC)(kernel='rbf', C=C, gamma=gamma))
            methods.append(delayed(sl.svm.SVC)(kernel='poly', degree=2, C=C,
                                               gamma=gamma))
            methods.append(delayed(sl.svm.SVC)(kernel='poly', degree=3, C=C,
                                               gamma=gamma))
        methods.append(delayed(sl.svm.SVC)(kernel='linear', C=C))

    # Misc.
    methods.append(delayed(sl.tree.DecisionTreeClassifier)())
    methods.append(delayed(sl.naive_bayes.GaussianNB)())
    """
    # Random Forests.
    for n in [10, 50, 100, 200, 500]:
        methods.append(delayed(sl.ensemble.RandomForestClassifier)
                              (n_estimators=n))

    for n in [10, 50, 100, 200, 500]:
        methods.append(delayed(sl.ensemble.RandomForestClassifier)
                              (n_estimators=n, criterion='entropy'))

    # Gradient boosting.
    for n in [10, 50, 100, 200, 500]:
        for a in [.0001, .001, .01, .1, 1., 10.]:
            methods.append(delayed(sl.ensemble.GradientBoostingClassifier)
                                  (n_estimators=n, learn_rate=a))

    # Nearest neighbors.
    for n in [1, 5, 10, 15, 20]:
        methods.append(delayed(sl.neighbors.KNeighborsClassifier)
                              (n_neighbors=n))

    for n in [1, 5, 10, 15, 20]:
        methods.append(delayed(sl.neighbors.KNeighborsClassifier)
                              (n_neighbors=n, weights='distance'))

    # l1-penalized logistic regression
    for c in [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]:
        methods.append(delayed(sl.linear_model.LogisticRegression)
                              (C=c, penalty='l1', tol=0.01))

    # l2-penalized logistic regression
    for c in [0.0001, 0.001, 0.01, 0.05, 0.1, 0.5, 1, 5, 10, 50, 100]:
        methods.append(delayed(sl.linear_model.LogisticRegression)
                              (C=c, penalty='l2', tol=0.01))


    return methods
