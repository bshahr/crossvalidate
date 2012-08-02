from __future__ import division
import numpy as np
import cPickle as pickle

__all__ = ['delayed', 'run_delayed']


def delayed(function):
    """
    Decorator returning a "delayed" version of the given function.

    This returns a wrapper around the given function such that calling the
    function will return a tuple (f, args, kwargs).

    This is a nearly direct copy of the function of the same name from joblib,
    mostly to add different documentation, and also to get rid of the call to
    functools.wraps.
    """
    # Try to pickle the input function, to catch the problems early when
    # using with multiprocessing (this is mostly just used by joblib).
    pickle.dumps(function)

    def delayed_function(*args, **kwargs):
        return function, args, kwargs

    return delayed_function


def run_delayed(delayed_function):
    """
    Run a "delayed" function (or class initialization) and return its result.
    """
    f, args, kwargs = delayed_function
    return f(*args, **kwargs)
