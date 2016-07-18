import numpy as np
import pandas as pd
from math import exp

# the softmax function must be stabilized against underflow and overflow

def softmax(x):
    out = np.zeros(len(x))
    den = 0
    for j in range(len(x)):
        den += exp(x[j]) 
    for i in range(len(x)):
        out[i] = exp(x[i])/den
    return(out)
    
c = -100
x = c*np.ones(10)
out = softmax(x)

# if we use c = -1000, then there is an error for float division by zero

# we can make the result stable to overflow by subtracting a scalar from the input vector 

def softmax_stable(x):
    z = x - max(x)
    out = softmax(z)
    return(out)

c = -1000
x = c*np.ones(10)
out = softmax_stable(x)

# Theano automatically detects and stabilizes many common numerically unstable expressions that arise in deep learning applications

a = np.random.random([10,10])
A = np.multiply(a,a.transpose())

def cond_numb(A):
    w, v = np.linalg.eig(A)
    out = abs(max(w)/min(w))
    return(out)
    
out = cond_numb(A)