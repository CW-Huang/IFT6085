# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 21:39:54 2017

@author: Chin-Wei
"""

import theano.tensor as T


def log_sum_exp(A, axis=None, sum_op=T.sum):

    A_max = T.max(A, axis=axis, keepdims=True)
    B = T.log(sum_op(T.exp(A - A_max), axis=axis, keepdims=True)) + A_max

    if axis is None:
        return B.dimshuffle(())  # collapse to scalar
    else:
        if not hasattr(axis, '__iter__'): axis = [axis]
        return B.dimshuffle([d for d in range(B.ndim) if d not in axis])  # drop summed axes

def log_mean_exp(A, axis=None):
    return log_sum_exp(A, axis, sum_op=T.mean)