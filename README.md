# Fast and Optimal Algorithms for Locally Private Mean Estimation

This software project accompanies the research paper, [Faster Optimal Locally Private Mean Estimation
via Random Projections](https://arxiv.org/abs/2306.04444).

We provide implementations for the algorithms in the above paper, along with code to reproduce all experiments. 


## Documentation

The main LDP mean estimation algorithms are implemented in the file PrivUnitAlgs.py. Code for reproducing the experiments can be found in experiments.py and experiments_MNIST.py for the general and MNIST experiments, respectively.



## Running Experiments

Download the repository and install all required packages as listed in requirements.txt.

To run the general experiment (synthetic data), simply run the file experiments.py:

```
python experiments.py
```

For the MNIST experiments, first run the script train_MNIST_script.sh to produce the results of private training, then run experiments_MNIST.py to produce the plots:

```
bash train_MNIST_script.sh
python experiments_MNIST.py
```
