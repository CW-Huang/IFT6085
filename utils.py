# -*- coding: utf-8 -*-
"""
Created on Fri Mar 31 21:39:54 2017

@author: Chin-Wei
"""

import theano.tensor as T
import numpy as np

c = - 0.5 * np.log(2*np.pi)

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
    
    
def log_stdnormal(x):
    return c - 0.5 * x**2 

def log_normal(x,mean,log_var,eps=0.0):
    return c - log_var/2 - (x - mean)**2 / (2 * T.exp(log_var) + eps)
    