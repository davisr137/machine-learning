#!/bin/python3

import numpy as np 
from math import log, exp

# http://rob.schapire.net/papers/explaining-adaboost.pdf

# define function for adaptive boosting (AdaBoost)
def AdaBoost(N,M):

    # randomly select -1 and +1 for response data
    y = np.random.random(M)
    Y = np.ones(M)
    Y[y >= 0.5] = 1
    Y[y < 0.5] = -1
    
    # add noise to predictors
    X = np.zeros([N,M])
    for i in range(N):
        x = np.zeros(M) + y*(0.001)
        X[i,:] = np.sign(x+np.random.normal(0,0.01,M))
    
    # initialize D, alpha, epsilon
    D = np.ones(M)/M
    alpha = np.zeros(N)
    epsilon = np.zeros(N)
    
    # compute alphas 
    for i in range(N):
        # get weak hypothesis
        weak = X[i,:]
        # get classification error
        epsilon[i] = 1-sum(weak == Y)/M
        # get scalar for loss function
        alpha[i] = 0.5*log((1-epsilon[i])/epsilon[i])
        # update D 
        for j in range(M):
            D[j] = D[j]*exp(-alpha[i]*Y[j]*weak[j])
        # regularize so D is a distribution 
        D = D/sum(D)
    
    # get boosted classifications
    boosted = np.zeros(M)
    for i in range(M):
        s = 0
        for j in range(N):
            s+=X[j,i]*alpha[j]
        boosted[i] = np.sign(s)
        
    # get naive classifications
    naive = np.zeros(M)
    for i in range(M):
        naive[i] = np.sign(sum(X[:,i]))
        
    epsilon_boost = 1-sum(boosted == Y)/M
    epsilon_naive = 1-sum(naive == Y)/M
    
    res = [epsilon_boost,epsilon_equal,alpha]
    return(res)
  
# test AdaBoost with synthetic data

M = 30000
N = 100
[epsilon_boost,epsilon_equal,alpha] = AdaBoost(N,M)



