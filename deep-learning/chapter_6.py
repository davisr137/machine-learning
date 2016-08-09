#!/bin/python3

import sys
import os 
import numpy as np
from math import factorial, floor, sqrt
from scipy.optimize import minimize

## Train linear model on XOR data

# data
X = [[0,0],[0,1],[1,0],[1,1]]
Y = [0,1,1,0]
X = np.array(X)
Y = np.array(Y)

# parameters
w = 1
b = 0.2
theta = [w,b]

# choose a linear model with theta consisting of w and b
def f(x,theta):
    w = theta[0]
    b = theta[1]
    out = sum(np.dot(x,w)) + b
    return(out)

# MSE loss function
def J(theta,X,Y):
    m = len(Y)
    loss = 0
    for i in range(m):
        loss += (Y[i] - f(X[i],theta))**2
    out = loss/m
    return(out)
    
# fit model using MSE; observe that w = 0 and b=  0.5
model = minimize(J, theta, args = (X,Y), method='nelder-mead',options={'xtol': 1e-8, 'disp': True})
     




    
