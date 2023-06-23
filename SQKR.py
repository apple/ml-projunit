# Most of the code from https://github.com/WeiNingChen/Kashin-mean-estimation/blob/master/privUnit_DJW_SQKR.ipynb

import math
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.linalg import hadamard
import scipy.io as io
import scipy.special as sc
from scipy.stats import ortho_group
import os
from utilities import *




# PrivHS randomizer: applies PrivHS to each row in X
def PrivHS(X,eps,k):
    X_t = X.T
    Y_t = Rep_DJW(X_t,eps,k)
    return Y_t.T

# SQKR randomizer: applies SQKR to each row in X
def SQKR(X,eps,k,FWHT=False):
    X_t = X.T
    Y_t = Kashin_full(X_t,eps,k,FWHT)
    return Y_t.T

# Applies DJW to the columns of the input matrix X
# Communication is 1-bit
def DJW(X, eps):

  (d, n) = X.shape
   # Using {d \choose d/2} approximation
  B = (math.exp(eps)+1)/(math.exp(eps)-1)*np.sqrt(math.pi*d/2)
  pi_a = math.exp(eps)/(1+math.exp(eps))
  X_perturb = X.copy()
    
  for i in range(n):
    # only handle when X[:, i] is a unit vector
    v = np.random.normal(0, 1, size = d)
    v = v/np.linalg.norm(v, 2) # v uniform over l_2 unit ball
    if np.sum(v * X[:, i]) < 0:
        v = -v
    T = 2*np.random.binomial(1, pi_a)-1
    X_perturb[:, i] = T*v
  
  return B*X_perturb


# Applies DJW k times with eps_i = eps/k
# Communicates k bits     
def Rep_DJW(X,eps,k):
    X_hat = np.zeros(X.shape)
    for i in range(k):
      X_perturb = DJW(X,1.0*eps/k)
      X_hat = X_hat + 1.0*X_perturb/k
    return X_hat




# Applies DJW k times with eps_i = eps/k
# Communicates k bits     
def Rep_DJW_single(x,eps,k):
    x_hat = np.zeros(len(x))
    for i in range(k):
      x_perturb = DJW_single(x,1.0*eps/k)
      x_hat = x_hat + 1.0*x_perturb/k
    return x_hat


def rand_quantize(a, a_bdd):
    return (np.random.binomial(1, (np.clip(a, -a_bdd, a_bdd)+a_bdd)/(2*a_bdd))-1/2)*2*a_bdd

def rand_sampling(q, k):
    # each column of q represents a quantized observation with Kashin representation
    # output k sampling matrices and an aggregation of q*sampling_mat
    (N, n) = q.shape
    sampling_mat_sum = np.zeros((n, N))
    sampling_mat_list = []
    for i in range(k):
        spl = np.eye(N)[np.random.choice(N, n)]
        sampling_mat_sum = sampling_mat_sum + spl 
        sampling_mat_list.append(spl.T)

    return [sampling_mat_list, sampling_mat_sum.T, q * sampling_mat_sum.T/k]

def kRR(k, eps, q_sampling, sampling_mat_list, a_bdd):
    # perturb each column of q, as a k-bit string, via k-RR mechanism 
    q_perturb = q_sampling.copy()
    (N, n) = q_sampling.shape
    for j in range(n):
        if (np.random.uniform(0,1) > (math.exp(eps)-1)/(math.exp(eps)+2**k-1)):
            noise = np.zeros(N)
            for i in range(k):
                # create a random {-1, +1}^N vector and filter it by sampling matrices
                noise = noise + (2*np.random.binomial(1, 1/2*np.ones(N))-1)*sampling_mat_list[i][:, j].reshape(-1,)/k

            q_perturb[:, j] = noise*a_bdd
    return q_perturb

def estimate(k, eps, q_perturb):
    return    (math.exp(eps)+2**k-1)/(math.exp(eps)-1)*q_perturb 

def kRR_string(d, num, eps):
    if (np.random.uniform(0,1) < math.exp(eps)/(math.exp(eps)+d-1)):
        return num
    else:
        return np.radom.choice(d)

# Y is nxm. add more zeros rows to make it n_big x m
def complete_with_zeros(y,n_big):
    (n,m) = y.shape
    if n == n_big:
        return y
    z = np.zeros((n_big,m))
    z[:len(y),:] = y
    return z
    
def Kashin_representation(x, U, FWHT=False, eta = 0.4, delta = 0.8):
    # compute kashin representation of x with respect to the frame U at level K
    d = x.shape[0]
    N = 2**int(math.ceil(math.log(d, 2))+1)
    a = np.zeros((N, 1))

    K = 1/((1-eta)*np.sqrt(delta)) # Set Kashin level to be K
    M = eta/np.sqrt(delta*N)

    y = x
    itr = int(np.log(N))
    for i in range(itr):
        if FWHT:
            b = applyHDs(complete_with_zeros(y,len(U)).squeeze(),None,[U])
        else:
            b = U @ y 
        b_hat = np.clip(b, -M, M)
        if FWHT:
            y = y - applyInverseHDs(b_hat,None,[U])[:d].reshape(y.shape)
        else:
            y = y - U.T @ b_hat
        a = a + b_hat.reshape(a.shape)
        M = eta*M
    if FWHT:
        b = applyHDs(complete_with_zeros(y,len(U)).squeeze(),None,[U]) 
        Ty = applyInverseHDs(b,None,[U])[:d]
    else:
        b = U @ y    
        Ty = U.T @ b 
    y = y - Ty
    a = a + b.reshape(a.shape)
    return [a, K/np.sqrt(N)]





def Kashin_encode(U, X, k, eps,FWHT=False):
    [a, a_bdd] = Kashin_representation(X, U,FWHT)
    q = rand_quantize(a, a_bdd)
    [sampling_mat_list, sampling_mat_sum, q_sampling] =  rand_sampling(q, k)
    q_perturb = kRR(k, eps, q_sampling, sampling_mat_list, a_bdd)
    return [q, q_sampling, q_perturb]

def Kashin_decode(U, k, eps, q_perturb,FWHT=False):
    if FWHT:
        N = len(U)
    else:
        (N, d) = U.shape
    q_unbiased = estimate(k, eps, q_perturb)
    if FWHT:
        return applyInverseHDs(complete_with_zeros(q_unbiased*N ,len(U)),None,[U])
    else:
        return U.T @ (q_unbiased*N)



def Kashin_full(X,eps,k,FWHT=False):
    d = X.shape[0]
    N = 2**int(math.ceil(math.log(d, 2))+1)
    if FWHT:
        U = np.random.choice(a=[-1, 1], size=(N), p=[0.5, 0.5]) # fast hadamard
        [q_quantize, q_sampling, q_perturb] = Kashin_encode(U, X, k, eps,True) #for Fast Hadamard
    else:
        U = ortho_group.rvs(dim=N).T[:, 0:d]
        [q_quantize, q_sampling, q_perturb] = Kashin_encode(U, X, k, eps)    
    X_hat = Kashin_decode(U, k, eps, q_perturb,FWHT)
    if FWHT:
        return X_hat[:d]
    return X_hat                  




        
        
        
        
        
        
        
        
        
        
        