#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
from scipy import stats as st
import math
from utilities import *
from SQKR import SQKR, PrivHS
import random

# Privatizes a vector x according to given mechanism
# x: input vector 
# eps: privacy parameter, if eps=-1 no privacy
# k: projection parameter for low-communication algorithms
# mech: in [PrivUnitG,ProjUnit,FastProjUnit,SQKR, PrivHS, RePrivHS,CompPrivUnitG]
# p,gamma,sigma: parameters for privG 
# fast: if True, use FWHT implementation of Hadamard transform which works
#       faster for the MNIST experiment
# W: transform to be used for corr-transform algorithms such as FastProjUnit-corr 
#    and ProjUnit-corr. 
#    For FastProjUnit-corr, one should use 
#    W = np.random.choice(a=[-1, 1], size=(n), p=[0.5, 0.5]) 
#    For ProjUnit-corr, one should use 
#    S = np.sort(random.sample(range(0, n), min(k,n))) #list of indices (different for different users)
#    W = W_full[S,:]
#
def privatize_vector(x,eps,k,mech,p=None,gamma=None,sigma=None,fast=False,W=None):
    n = len(x)
    if mech in ['PrivUnitG', 'ProjUnit', 'FastProjUnit']: 
        if p is None or  gamma is None or  sigma is None:
            # parameters for PrivG algorthms
            p = priv_unit_G_get_p(eps)
            gamma, sigma = get_gamma_sigma(p, eps)
    if mech == 'PrivUnit':
        return PrivUnit(x,eps)
    elif mech == 'PrivUnitG':
        return PrivUnitG(x,eps,p,gamma,sigma)
    elif mech == 'CompPrivUnitG':
        return CompPrivUnitG(x, eps, p, gamma, sigma)
        return priv_G_compressed(x, eps, p, gamma, sigma)
    elif mech == 'FastProjUnit':
        return FastProjUnit(x,eps,k,p,gamma,sigma,fast)
    elif mech == 'FastProjUnit-corr':
        if W is None:
            assert("Base transform not provided")
        return FastProjUnit(x,eps,k,p,gamma,sigma,fast,W)
    elif mech == 'ProjUnit':
        return ProjUnit(x,eps,k,p,gamma,sigma)
    elif mech == 'ProjUnit-corr':
        if W is None:
            assert("corr transform not provided")
        return ProjUnit(x,eps,k,p,gamma,sigma,W)
    elif mech == 'RePrivHS':
        return PrivHS(x.reshape(1,len(x)),eps,k).squeeze() 
    elif mech == 'PrivHS':
        k = 1
        return PrivHS(x.reshape(1,len(x)),eps,k).squeeze() 
    elif mech == 'SQKR':
        return SQKR(x.reshape(1,len(x)),eps,k,True).squeeze()
    



# Applies the ProjUnit algorithm based on Gaussian projections 
# for input vector x
# x: input vecotr 
# eps: privacy parameter, if eps=-1 no privacy
# p,gamma,sigma: parameters for privG 
# R_p: predefined projection matrix. If None, sample fresh rotation matrix
def ProjUnit(x,eps,k,p=None,gamma=None,sigma=None,R_p=None):
  if p is None or gamma is None or sigma is None:
    p = priv_unit_G_get_p(eps)
    gamma, sigma = get_gamma_sigma(p, eps)
  n = len(x)
  
  R = None
  if R_p is None:
      vectors = np.random.rand(k, n)
      q, _ = np.linalg.qr(vectors.T)
      R = math.sqrt(1.0*n/k) * q.T
  else:
      R = R_p
  clipped_Rx = normalize (R @ x)
  noisy_Rx = None
  if eps == -1:
    noisy_Rx = clipped_Rx
  else:
    noisy_Rx = PrivUnitG(clipped_Rx, eps, p, gamma, sigma)

  z = np.transpose(R) @ noisy_Rx
  return z



# Applies the FastProjUnit algorithm based on the SRHT transform
# for an input vector x
# x: input vector with unit norn
# eps: privacy parameter, if eps=-1 no privacy
# k: projection to k-dimensional sub-space
# p,gamma,sigma: parameters for privG 
# D: predefined D for the HD transform. If none, sample new D
def FastProjUnit(x,eps,k,p=None,gamma=None,sigma=None,fast=False,D=None):
  n = len(x)
  if p is None or gamma is None or sigma is None:
    p = priv_unit_G_get_p(eps)
    gamma, sigma = get_gamma_sigma(p, eps)
  
  D1 = None
  if D is None:
     D1 = np.random.choice(a=[-1, 1], size=(n), p=[0.5, 0.5])
  else:
     D1 = D
  #S = np.sort(random.choices(range(n), k=k)) # with repitition
  S = random.sample(range(n), k) # without repitition
  clipped_z = normalize(applyHDs(x, S, [D1],fast))
  noisy_z = None
  if eps == -1:
    noisy_z = clipped_z
  else:
    noisy_z = PrivUnitG(clipped_z, eps, p, gamma, sigma)
  z = applyInverseHDs(noisy_z, S, [D1],fast)
  return z



# Applies the PrivUnitG randomizer over input vector x
# x: input vector with unit norm
# eps: privacy parameter, if eps=-1 no privacy
# p,gamma,sigma: parameters for privG 
# n_trials: number of trials for sampling for the tail of gaussian
def PrivUnitG(x, eps, p=None, gamma=None, sigma=None, n_tries=None):
    if not p:
        p = priv_unit_G_get_p(eps)
    if p is None or gamma is None or sigma is None:
        gamma, sigma = get_gamma_sigma(p, eps)
    dim = x.size
    g = np.random.normal(0, 1, size = dim)
    pos_cor = np.random.binomial(1, p)

    if pos_cor:
        chosen_dps = np.array([sample_from_G_tail_stable(gamma)])
    else:
        if n_tries is None:
          n_tries = 25 # here probability of success is 1/2
        dps = np.random.normal(0, 1, size=n_tries)
        chosen_dps = dps[dps<gamma]
    
    if chosen_dps.size == 0:
        print('failure')
        return g * sigma
    target_dp = chosen_dps[0]

   
    yperp = g - (g.dot(x)) * x
    ypar = target_dp * x
    return sigma * (yperp + ypar)


# Applies the CompressedPrivUnitG randomizer over input vector x
# x: input vector with unit norn
# eps: privacy parameter, if eps=-1 no privacy
# p,gamma,sigma: parameters for privG 
# n_trials: number of trials for sampling for the tail of gaussian
def CompPrivUnitG(x, eps, p=None, gamma=None, sigma=None, n_tries=None):
    # First get p, gamma, sigma
    if not p:
        p = priv_unit_G_get_p(eps)
    if not gamma or not sigma:
        gamma, sigma = get_gamma_sigma(p, eps)
    dim = x.size
    if n_tries is None:
        n_tries = math.ceil(math.exp(eps) * math.log(1e4))

    
    # We will sample up to n_tries iid Gaussians, and do rejection sampling
    # The selection probabilty at any step is p/q(gamma)exp(eps0) if sample is above gamma
    #   and (1-p)/(1-q(gamma))exp(eps0) if sample is below gamma. q(gamma) is as in the above get_p function
    
    qg = st.norm.sf(gamma)
    sp_high = p / (qg * math.exp(eps))
    sp_low = (1-p) / ((1-qg) * math.exp(eps))
    
    # Below is Algorithm 1 from https://arxiv.org/pdf/2102.12099.pdf
    for _ in range(n_tries):
        
        # In an actual compressed representation, the seed for this random g is what 
        # is actually sent. So we will use a seeded prg here. In a simulation, that is
        # an un-necessary overhead.
        g = np.random.normal(0, 1, size = dim)
        
        dp = g.dot(x)
        
        if dp > gamma:
            # In this case, we select with probability sp_high
            if np.random.binomial(1, sp_high) == 1:
                break
        else:
            # In this case, we select with probability 1-sp_low
            if np.random.binomial(1, sp_low) == 1:
                break
        
    # In simulation, we send sigma*g. In a compressed version, we send the seed
    # and the server regenerates (this same) g from the seed and uses sigma * g
    return sigma * g

