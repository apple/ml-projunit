#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy
from scipy.special import gamma
from scipy import stats as st
import math
from scipy.stats import norm
import math
import os


#normalize vector x to have norm 1
def normalize(x, ord=None): 
  return x / np.linalg.norm(x, ord)


# default is l2 clipping, but can also do l_p clipping by setting ord=p
def clip(x, ord=None): 
  return x / max(1, np.linalg.norm(x, ord))

# Adds 1 bit to complete the squared norm to C then normalizes to 1
def complete_then_normalize(x,C):
    b = max(C - np.linalg.norm(x)**2, 0)
    u = np.zeros(len(x)+1)
    u[:-1] = x
    u[-1] = math.sqrt(b)
    return normalize(u)


# Applies hadamard transform over x (without normalization)
# if n=len(x) is not a power of 2, we concatenate 0's to x
# so that len(x) is a power of 2 then apply the transform
def unnormalized_hadamard_transform(x):
  n = len(x)
  if n == 1:
    return x.copy()      
  a = math.log(n,2)
  if a - int(a) > 0:
      n_new = 2**(math.ceil(a))
      y1 = unnormalized_hadamard_transform(x[:n_new//2])
      x_new = np.zeros(n_new//2) # fill right side with zeros
      x_new[:n-n_new//2] = x[n_new//2:]
      y2 = unnormalized_hadamard_transform(x_new)
      ret = np.zeros(n_new)
      ret[:len(ret)//2] = y1 + y2
      ret[len(ret)//2:] = y1 - y2
      return ret[:n]
  else: 
      y1 = unnormalized_hadamard_transform(x[:len(x)//2])
      y2 = unnormalized_hadamard_transform(x[len(x)//2:])
      ret = np.zeros(len(x))
      ret[:len(ret)//2] = y1 + y2
      ret[len(ret)//2:] = y1 - y2
      return ret


# Applies the hadamard transform for a vector x
# if fast is True (for MNIST experiment), we apply FWHT which is 
# faster when d~60k
def hadamard_transform(x,fast):
  if fast:
      return FWHT(x)
  else:
      return unnormalized_hadamard_transform(x) / len(x)**0.5


def FWHT(x):
    n = len(x)
    a = math.log(n,2)
    if a - int(a) > 0:
        n_new = 2**(math.ceil(a))
        x_new = np.zeros(n_new) # fill with zeros
        x_new[:n] = x
        return unnormalized_FWHT(x_new)[:n]/ n**0.5
    else:
        return unnormalized_FWHT(x) / n**0.5       

# Faster hadamard implementation for lower-dimensional problems (d~60k)
# Code from Github Repo (https://github.com/dingluo/fwht) 
def unnormalized_FWHT(x,find_perm = True):
    """ Fast Walsh-Hadamard Transform
    Based on mex function written by Chengbo Li@Rice Uni for his TVAL3 algorithm.
    His code is according to the K.G. Beauchamp's book -- Applications of Walsh and Related Functions.
    """ 
    x_original = x
    x = x.copy()
    x = x.squeeze()
    N = x.size
    G = int(N/2) # Number of Groups
    M = 2 # Number of Members in Each Group

    # First stage
    y = np.zeros((int(N/2),2))
    y[:,0] = x[0::2] + x[1::2]
    y[:,1] = x[0::2] - x[1::2]
    x = y.copy()
    # Second and further stage
    for nStage in range(2,int(math.log(N,2))+1):
        y = np.zeros((int(G/2),M*2))
        y[0:int(G/2),0:M*2:4] = x[0:G:2,0:M:2] + x[1:G:2,0:M:2]
        y[0:int(G/2),1:M*2:4] = x[0:G:2,0:M:2] - x[1:G:2,0:M:2]
        y[0:int(G/2),2:M*2:4] = x[0:G:2,1:M:2] - x[1:G:2,1:M:2]
        y[0:int(G/2),3:M*2:4] = x[0:G:2,1:M:2] + x[1:G:2,1:M:2]
        x = y.copy()
        G = int(G/2)
        M = M*2
    x = y[0,:]
    x = x.reshape((x.size,1))
    ret = 1.0*x[:,0]
    # This functions returns a shuffles version of our hadamard-transform
    # Here, we find this permutation and return the exact vector as our
    # function unnormalized_hadamard_transform above
    if find_perm:
        perm = find_FWHT_perm(len(x))
        return ret[perm][0,:]
    else: 
        return ret


# Find a permutation that makes the output of unnormalized_FWHT and 
# unnormalized_hadamard_transform the same
def find_FWHT_perm(n):
    x = np.random.normal(0,1,n)
    f_name = 'dict_perm.npy'
    dict_perm = {}
    if os.path.exists(f_name):
        dict_perm = np.load(f_name,allow_pickle='TRUE').item()
    if n in dict_perm:
        perm = dict_perm[n]
    else:
        ret_right = unnormalized_hadamard_transform(x)
        ret_FWHT = unnormalized_FWHT(x,False)
        perm = find_perm(ret_right,ret_FWHT)
        dict_perm[len(x)] = perm
        np.save(f_name, dict_perm)  
   

# Input y is a permutation of w. 
# Returns a permutation such that w[perm]=y
def find_perm(y,w):
    #print(y)
    #print(w)
    perm = []
    for j in range(len(y)):
        perm.append(np.where(w==y[j])[0][0])
    perm = [int(p) for p in perm]
    return perm




# Applies HDS to vector x 
def applyHDs(x, S, D,fast=False):
  z = x.copy()
  for d in D:
    z = hadamard_transform(d*z,fast)
  if S is None:
    return z
  L = []
  for j in S:
    L.append(z[j])
  return (len(x) / len(S))**0.5 * np.array(L)

# Applies Inverse HDS to vector x 
def applyInverseHDs(x, S, D,fast=False):  
  z = np.zeros(len(D[0]))
  if S is None:
    z = x.copy()
    for d in D[::-1]:
      z = d*hadamard_transform(z,fast)
    return z
  else:
    for i in range(len(S)):
      z[S[i]] += x[i]
    for d in D[::-1]:
      z = d*hadamard_transform(z,fast)
    return (len(D[0]) / len(S))**0.5 * z


def get_gamma_sigma(p, eps):
    # Want p(1-q)/q(1-p) = exp(eps)
    # I.e q^{-1} -1 = (1-q)/q = exp(eps) * (1-p)/p
    qinv = 1 + (math.exp(eps) * (1.0-p)/ p)
    q = 1.0 / qinv
    gamma = st.norm.isf(q)
    # Now the expected dot product is (1-p)*E[N(0,1)|<gamma] + pE[N(0,1)|>gamma]
    # These conditional expectations are given by pdf(gamma)/cdf(gamma) and pdf(gamma)/sf(gamma)
    unnorm_mu = st.norm.pdf(gamma) * (-(1.0-p)/st.norm.cdf(gamma) + p/st.norm.sf(gamma))
    sigma = 1./unnorm_mu
    return gamma, sigma

def priv_unit_G_get_p(eps, return_sigma=False):
    # Mechanism:
    # With probability p, sample a Gaussian conditioned on g.x \geq gamma
    # With probability (1-p), sample conditioned on g.x \leq gamma
    # Scale g appropriately to get the expectation right
    # Let q(gamma) = Pr[g.x \geq gamma] = Pr[N(0,1) \geq gamma] = st.norm.sf(gamma)
    # Then density for x above threshold = p(x)  * p/q(gamma)
    # And density for x below threhsold = p(x) * (1-p)/(1-q(gamma))
    # Thus for a p, gamma is determined by the privacy constraint.
    plist = np.arange(0.01, 1.0, 0.01)
    glist = []
    slist = []
    for p in plist:
        gamma, sigma = get_gamma_sigma(p, eps)
        # thus we have to scale this rv by sigma to get it to be unbiased
        # The variance proxy is then d sigma^2
        slist.append(sigma)
        glist.append(gamma)
    ii = np.argmin(slist)
    if return_sigma:
        return plist[ii], slist[ii]
    else:
        return plist[ii]
    
# calculates MSE norm of PrivUnitG
def priv_unit_G_sq_norm(dim, eps):
    p, sigma = priv_unit_G_get_p(eps, return_sigma=True)
    var = dim * sigma * sigma
    return var


# Fast algorithm for sampling from a truncated Gaussian (conditioned on having value at least gamma)
# This combination of sf and isf seems to be the most stable one
# If we could get isf(exp(a)) , i.e. isf with input being in log space, then this can be made a lot more stable by doing q and r in log space
# This seems to work until gamma about 35 at least.

def sample_from_G_tail(gamma):
  q = norm.sf(gamma)
  r = np.random.uniform(low=0, high=q)
  #print(q,r)
  return norm.isf(r)

# More stable version. Works at least until 1000
def sample_from_G_tail_stable(gamma):
  return sample_from_G_tail(gamma)
  logq = norm.logsf(gamma)
  u = np.random.uniform(low=0, high=1)
  logu = np.log(u)
  logr = logq + logu # r is now uniform in (0,q)
  #print(q,r)
  return -scipy.ndtri_exp(logr)


