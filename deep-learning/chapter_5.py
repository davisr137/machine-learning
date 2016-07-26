#!/bin/python3

import sys
import os 
import numpy as np
from math import factorial, floor, sqrt
from scipy.optimize import minimize

# generate true polynomial of degree 2
# fit using polynomial of degree 9
# vary the regularization parameter to get various fits

# generate predictors
r = 10000
max_degree = 7
rv = np.random.random([r,1])
A = np.zeros([r,max_degree])
for i in range(r):
    for j in range(max_degree):
        power = max_degree-j-1
        A[i,j] = rv[i]**power

# x is the true value of the parameters
x = np.array([0,0,0,0,0.7,1.2,0.3])
# b is a vector of outputs
b = np.dot(A,x)
# add noise to the outputs
b = b + np.random.normal(0,0.05,r)

# define loss functions
def loss(xi):
    mse = np.linalg.norm(np.dot(A,xi) - b)
    reg = lmbda * np.dot(xi.transpose(),xi)
    return(mse + reg.item())
    
# fit with no regularization
x_est = np.ones([7,1])
lmbda = 0
res1 = minimize(loss,x_est,options={'disp': True})

# fit with regularization
lmbda = 10
res2 = minimize(loss,x_est,options={'disp': True})







    
