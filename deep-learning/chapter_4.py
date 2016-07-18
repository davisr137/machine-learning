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

# linear least squares example
# A*x = b

r = 1000
c = 5
a = np.random.random([r,c])
x_true = np.array([0.8,0.9,1,1.1,1.2])
b = np.zeros(r)
for i in range(r):
    s = 0
    for j in range(c):
        s += a[i,j]*x_true[j]
    b[i] = s
    
# add noise to predictors
A = a + np.random.normal(0,0.1,[r,c])

# estimate using numpy 
out = np.linalg.lstsq(A,b)
x1 = out[0]

# solve using gradient based optimization

def lsq_grad_desc(A,b,delta,epsilon):
    [row,col] = A.shape
    x = np.matrix(np.ones(col))
    x = x.transpose()
    grad = A.transpose() * A * x - A.transpose() * b
    while np.linalg.norm(grad) > delta:
        grad = A.transpose() * A * x - A.transpose() * b
        x = x - epsilon*grad
    return(x)
    
delta = 0.0001
epsilon = 0.001

A = np.matrix(A)
b = np.matrix(b)
b = b.transpose()

# observe that we get the same result with gradient descent as with the numpy library routine 
x2 = lsq_grad_desc(A,b,delta,epsilon)

    
        

