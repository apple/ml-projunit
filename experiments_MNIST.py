# Code for MNIST experiment

import numpy as np
import matplotlib.pyplot as plt
import os
import re






def color_plot(mech):
    if 'PrivUnitG' in mech:
        return 'k'
    elif 'CompPrivUnitG' in mech :
        return 'y'
    elif 'FastProjUnit-corr' in mech :
        return 'm'
    elif 'FastProjUnit' in mech :
        return 'b'
    elif 'ProjUnit' in mech:
        return 'r'
    elif 'RePrivHS' in mech :
        return 'g'
    elif 'PrivHS' in mech:
        return 'c'
    elif 'SQKR' in mech:
        return 'm'
    elif 'Gaussian' in mech:
        return 'olive'
    elif 'nonPrivate' in mech:
        return 'g'
    return 'k'


# This function assumes that the folder dpsgd_results contains 
# the data for the various methods. 
# Then we plot the test error as a function of epoch for all methods 
#in this folder that has privacy parameter eps and number of epochs 
def plot_eps(eps,epochs,dataset="mnist"):
    f_dir = 'MNIST_results'
    plt.figure()
    c_mech = 0 #counter for marker of PrivHS
    colors_phs = ['c', 'r', 'm'] #colors for phs
    mechanisms = ['Gaussian', 'PrivUnitG', 'FastProjUnit', 'FastProjUnit-corr', 'RePrivHS']
    for mech in mechanisms:
        for f_name in os.listdir(f_dir):
            f = f_dir + '/' + f_name
            if f_name[-3:] != 'npy' or 'num_rep' not in f_name:
                continue
            f_eps = int(float(re.findall('_.*.',re.findall('epsilon_.*.pth', f_name)[0])[0][1:-4]))
            f_epochs = int(re.findall('_.*_',re.findall('epochs_.*_lr', f_name)[0])[0][1:-1])
            f_mech = re.findall('_.*_',re.findall('mech_.*_num', f_name)[0])[0][1:-1]
            if mech != f_mech:
                continue
            f_lr = float(re.findall('_.*_',re.findall('lr_.*_clip', f_name)[0])[0][1:-1])
            if f_eps == eps and f_epochs == epochs:
                res = np.load(f)
                q = 0.9
                err_mean = [np.mean(res[i,:]) for i in range(res.shape[0])]
                err_high = [np.quantile(res[i,:],q) for i in range(res.shape[0])]
                err_low = [np.quantile(res[i,:],1-q) for i in range(res.shape[0])]
                f_k = 0
                if f_mech in ['ProjUnit', 'FastProjUnit', 'FastProjUnit-corr','RePrivHS', 'SQKR']:
                    # find communication k
                    f_k = f_epochs = int(re.findall('_.*_',re.findall('k_.*_epsilon', f_name)[0])[0][1:-1])
                    if f_mech == 'RePrivHS':
                        f_mech = f_mech + ' (R = %d)' % (f_k)
                    else:
                        f_mech = f_mech + ' (k = %d)' % (f_k)
                
                if 'RePrivHS' in f_mech:
                    c = colors_phs[c_mech]
                    c_mech = c_mech + 1
                    plt.plot(range(epochs),err_mean,color=c, marker='o',label=f_mech)
                    plt.fill_between(range(epochs), err_low, err_high, color=c, alpha=.1)
                else:
                    plt.plot(range(epochs),err_mean,color=color_plot(f_mech), marker='o',label=f_mech)
                    plt.fill_between(range(epochs), err_low, err_high, color=color_plot(f_mech), alpha=.1)
                    
    plt.title("%s (epsilon = %d)" % (dataset.upper(),int(eps)))
    plt.legend()
    plot_name = "plots/%s_eps_%d_epochs_%d.pdf" % (dataset,int(eps),epochs)
    plt.savefig(plot_name)
    plt.show() 




# Before running this, you have to run the script train_MNIST_script.sh
# to produce the results for all methods.
# This might take some time. You can also run other setting
# by  using the following command:
# python train_mnist.py --epsilon 10 --epochs 10 --lr 0.1 --clip-val 1 --mechanism FastProjUnit --k 1000



# Plot the main results

# Experiment 1: eps=4 and epochs=10
eps = 4
epochs = 10
plot_eps(eps,epochs)

# Experiment 2: eps=10 and epochs=10
eps = 10
epochs = 10
plot_eps(eps,epochs)


# Experiment 3: eps=16 and epochs=10
eps = 16
epochs = 10
plot_eps(eps,epochs)






