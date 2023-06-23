#!/usr/bin/env python3
# -*- coding: utf-8 -*-


import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from utilities import *
from PrivUnitAlgs import *
import time
from SQKR import *
import os
import time
import signal

# A function that takes as input a mechanism mech and 
# privacy parameter epsilon, and returns the "best" k 
# for this setting. Best here is based on our simulations for 
# each algorithm with different values of epsilon and k
def choose_best_k(mech,eps):
    k = int(eps)
    if mech in ['ProjUnit', 'FastProjUnit']:
        k = 1000
    elif mech == 'SQKR':
        k = int(eps)
    elif mech == 'PrivHS':
        k = 1
    elif mech == 'RePrivHS':
        k = max(int(eps/2),1)
    return k


# returns (color,mark) for mechnism
def color_plot(mech):
    if mech == 'PrivUnitG':
        return ('k','o')
    if mech == 'CompPrivUnitG':
        return ('y', 'o')
    elif mech == 'ProjUnit':
        return ('r','>')
    elif mech == 'ProjUnit-corr':
        return ('c','o')
    elif mech == 'FastProjUnit':
        return ('b','<')
    elif mech == 'FastProjUnit-corr':
        return ('g','x')
    elif mech == 'RePrivHS':
        return ('g','o')
    elif mech == 'PrivHS':
        return ('c','o')
    elif mech == 'SQKR':
        return ('m','o')
    return ('k','0')




# Takes as input a vector x and epsilon, and calculates the error 
# of the mechanism for num_rep trials and returns an array with the MSEs.
# The calculation is done as follows: we run the mechanism mech
# over input x for num_rep times, recording the MSE of each trial
# and returning the mean of these MSEs.
# x: input vector
# eps: privacy parameter
# mech: a mechanism to run
# num_rep: num of repetitions to run the mechanism
def find_err(x,eps,k,mech,num_rep):
    y = np.zeros(len(x))
    p = None,
    gamma = None
    sigma = None
    if mech in ['PrivUnitG', 'ProjUnit', 'FastProjUnit', 'ProjUnit-corr','FastProjUnit-corr']: 
        # parameters for PrivG algorthms
        p = priv_unit_G_get_p(eps)
        gamma, sigma = get_gamma_sigma(p, eps)
    err_mech_arr = np.zeros(num_rep)
    for i in range(num_rep):
        y = privatize_vector(x,eps,k,mech,p,gamma,sigma)
        err_mech_arr[i] = np.sum((y-x)**2)
    return err_mech_arr

    
    
    
    
# Plots the variance as a function of epsilon for all methods
# For each method, k is chosen using the function choose_best_k.
def exp_compare_var_eps(num_rep = 100):
    mechanisms = [ 'FastProjUnit', 'ProjUnit',  'PrivUnitG',  'PrivHS','RePrivHS','SQKR']
    n = 2**15 # dimension
    lst_eps = list(range(16))[1:16:2] 
    
    err_dict = {}
    dict_file = 'raw/err_comp_eps_dim_%d_num_rep_%d.npy' % (n,num_rep)
    if os.path.exists(dict_file):
        err_dict = np.load(dict_file,allow_pickle='TRUE').item()
    x = np.random.normal(size=n)
    x = x / np.linalg.norm(x)
    for mech in mechanisms:
        print(mech)
        if mech in err_dict:
            continue
        err_dict[mech] = np.zeros((len(lst_eps),num_rep))
        for i in range(len(lst_eps)):
            eps = lst_eps[i]
            k = choose_best_k(mech,eps)
            err_dict[mech][i] = find_err(x,eps,k,mech,num_rep)
                
        np.save(dict_file, err_dict) 
    
    # plot the errors
    plt.figure()
    for mech in mechanisms:
        (c,m) = color_plot(mech)
        q = 0.9
        err_mech = [np.mean(err_dict[mech][i]) for i in range(len(lst_eps))]
        err_mech_high = [np.quantile(err_dict[mech][i],q) for i in range(len(lst_eps))]
        err_mech_low = [np.quantile(err_dict[mech][i],1-q) for i in range(len(lst_eps))]
        (c,m) = color_plot(mech)
        plt.plot(lst_eps,err_mech,color = c, marker=m,linestyle='dashed',label = mech)
        plt.fill_between(lst_eps, err_mech_low, err_mech_high, color=c, alpha=.1)
        
        
    plt.legend()
    plt.yscale("log")
    f_name = 'plots/err-comparison_lst_eps_d_%d.pdf' % n
    plt.savefig(f_name)
    plt.show()     
        
  
# Plot the variance as a function of k for all methods
# Here we use the same HD matrices for the ProjUnit algorithms
# and only sample different S_i matrix for each user
# num_users: number of users
# num_rep: number of repetitions 
def exp_compare_algs_k(eps,num_users=20,num_rep=100):
    mechanisms = ['PrivUnitG', 'ProjUnit', 'FastProjUnit', 'FastProjUnit-corr'] 
    n = 2**13 # dimension of input vectors
    lst_k = [10,  50, 100,  500, 800, 1000, 1500, 2000] 
    f_name = 'plots/err-fixHD_num_users_%d_num_rep_%d_d_%d_eps_%d.pdf' % (num_users,num_rep,n,int(eps))
    dict_file = 'raw/err-fixHD_lst_k_num_users_%d_num_rep_%d_d_%d_eps_%d.npy' % (num_users,num_rep,n,int(eps))
    err_dict = {}
    if os.path.exists(dict_file):
        err_dict = np.load(dict_file,allow_pickle='TRUE').item()
    
    p = priv_unit_G_get_p(eps)
    gamma, sigma = get_gamma_sigma(p, eps)
    
    v = np.random.normal(size=n)
    v = v/math.sqrt(np.sum(v**2))
    X = np.random.normal(size=(num_users,n))/math.sqrt(n) + v
    norms = np.linalg.norm(X, axis=1)
    X = X / norms[:, np.newaxis]
    norms = np.linalg.norm(X, axis=1)
    v_hat = np.sum(X,0)/num_users
    for mech in mechanisms:
        print(mech)
        if mech in err_dict:
            continue
        err_dict[mech] = {}
        for i in range(len(lst_k)):
            k = lst_k[i]
            err_dict[mech][k] = np.zeros(num_rep)
            if mech == 'PrivUnitG' and i>0:
                err_dict[mech][k] = err_dict[mech][lst_k[0]]
                continue
            err = np.zeros(num_rep)
            W = None
            for j in range(num_rep):
                x_hat = np.zeros(n)
                if mech == 'FastProjUnit-corr':
                    W = np.random.choice(a=[-1, 1], size=(n), p=[0.5, 0.5]) 
                elif mech == 'ProjUnit-corr':
                    W_full =  special_ortho_group.rvs(n)
                for u in range(num_users):
                    if mech == 'ProjUnit-corr':
                        S = np.sort(random.choices(range(n), k=k)) # without repitition
                        #S = random.sample(range(n), k)  # with repitition
                        W = math.sqrt(1.0*n/k) * W_full[S,:]
                    x_hat += privatize_vector(X[u],eps,k,mech,p,gamma,sigma,False,W)/num_users
                err[j] = np.sum((x_hat - v_hat)**2)
            err_dict[mech][k] = err
                
    np.save(dict_file, err_dict)  


    # plot the errors
    plt.figure()
    for mech in mechanisms:
        q = 0.9
        err_mech = [np.mean(err_dict[mech][k]) for k in lst_k]
        err_mech_high = [np.quantile(err_dict[mech][k],q) for k in lst_k]
        err_mech_low = [np.quantile(err_dict[mech][k],1-q) for k in lst_k]
        (c,m) = color_plot(mech)
        plt.plot(lst_k[1:],err_mech[1:],color = c, marker=m,linestyle='dashed',label = mech)
        plt.fill_between(lst_k[1:], err_mech_low[1:], err_mech_high[1:], color=c, alpha=.1)
        
    #plt.xlabel('k')
    plt.legend()
    plt.yscale("log")
    plt.xscale("log")
    plt.savefig(f_name)
    plt.show()     
    


def handle_timeout(signum, frame):
    raise TimeoutError


        

# Plot the run-time of each method as a function of the dimension
# when k is chosen as the best k for each method (using choose_best_k).
# For each dimension, we give a cut-off of 1 hour for each method 
# to finish (num_rep=10) trials
def experiment_compare_time(eps,num_rep = 10):
    # Run the function to create the plots
    mechanisms = ['PrivUnitG', 'ProjUnit', 'FastProjUnit', 'PrivHS', 'SQKR', 'CompPrivUnitG']
    
    file_dict = 'raw/time_dict_eps_%d.npy' % int(eps)
    time_dict = {}
    if os.path.exists(file_dict):
        time_dict = np.load(file_dict,allow_pickle='TRUE').item()
    
    
    lst_n = [10, 10**2, 10**3, 10**4, 10**5, 10**6, 10**7]
    
    time_arr = {}
    
    p = None
    gamma = None
    sigma = None
    time_limit = 60*60 # set time limit to 60 minutes
    for mech in mechanisms:
        print(mech)
        time_arr[mech] = np.zeros(len(lst_n))
        if mech in ['PrivUnitG', 'FastProjUnit', 'ProjUnit', 'CompPrivUnitG']:
            p = priv_unit_G_get_p(eps)
            gamma, sigma = get_gamma_sigma(p, eps)
        for i in range(len(lst_n)):
            n = lst_n[i]
            if (mech,n) in time_dict:
                time_arr[mech][i] = time_dict[mech,n]
                continue
            x = np.random.normal(size=n)
            x = x / np.linalg.norm(x)
            k = choose_best_k(mech,eps)
            
            start_time = time.process_time()
            signal.signal(signal.SIGALRM, handle_timeout)
            if i > 0 and time_arr[mech][i-1] == time_limit:
                time_arr[mech][i] = time_limit
                time_dict[mech,n] = time_limit
                continue
            signal.alarm(time_limit)  # give limited time to each application
            try:
                for j in range(num_rep):  
                    y = privatize_vector(x,eps,k,mech,p,gamma,sigma) 
                time_arr[mech][i] = time.process_time() - start_time
            except TimeoutError:
                print("------Too long-------")
                time_arr[mech][i] = time_limit
            
            time_dict[mech,n] =  time_arr[mech][i]


    np.save(file_dict, time_dict)  


    # plot the results
    plt.figure()
    for mech in mechanisms:
        if time_arr[mech][-1] == time_limit:
            idx = np.where(time_arr[mech] == time_limit)[0][0]
        else:
            idx = -1
        if mech == 'PrivUnitG':
            plt.plot(lst_n[:idx],time_arr[mech][:idx],'k-o',label = mech)
            continue
        plt.plot(lst_n[:idx],time_arr[mech][:idx],color_plot(mech),label = mech)
    plt.legend()
    plt.xscale("log")
    plt.yscale("log")
    f_name = 'plots/time-comparison_eps_%d.pdf' % int(eps)
    plt.savefig(f_name)
    plt.show()     
        





# ---------- Experiment 1: plot error as a function of k  ---------
num_rep = 30
num_users = 50

print('----experiment 1 (eps = 16)------')
eps = 16.0
exp_compare_algs_k(eps,num_users,num_rep)


print('----experiment 1 (eps = 10)------')
eps = 10.0
exp_compare_algs_k(eps,num_users,num_rep)


print('----experiment 1 (eps = 4)------')
eps = 4.0
exp_compare_algs_k(eps,num_users,num_rep)




# ---------- Experiment 2: plot error as a function of epsilon ---------
num_rep = 50

print('----experiment 2------')
exp_compare_var_eps(num_rep)





# ---------- Experiment 3: plot time for each method  ---------
num_rep = 10

print('----experiment 3 (eps = 10)------')
eps = 10
experiment_compare_time(eps,num_rep)


print('----experiment 3 (eps = 16)------')
eps = 16
experiment_compare_time(eps,num_rep)














