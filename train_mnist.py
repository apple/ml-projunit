# Based on Opacus code (https://github.com/pytorch/opacus/blob/main/examples/mnist.py)


import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from opacus import PrivacyEngine
from opacus.grad_sample.grad_sample_module import GradSampleModule
from torchvision import datasets, transforms
from tqdm import tqdm

from PrivUnitAlgs import *
from gaussian_analytic import calibrateAnalyticGaussianMechanism
from time import sleep


# Precomputed characteristics of the MNIST dataset
MNIST_MEAN = 0.1307
MNIST_STD = 0.3081


# Model from https://arxiv.org/abs/2203.08134
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 8, 2, padding=2)
        self.conv2 = nn.Conv2d(16, 64, 4, 2, padding=0)
        self.fc1 = nn.Linear(64 * 5 * 5, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = x.view(-1, 1, 28, 28)
        x = self.conv1(x)
        x = torch.tanh(x)
        x = F.avg_pool2d(x, 1)
        x = self.conv2(x)
        x = torch.tanh(x)
        x = F.avg_pool2d(x, 1)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = torch.tanh(x)
        x = self.fc2(x)
        return x
    


# Convert gradients into vectors
def grad_to_vec(model):
    res = []
    for p in model.parameters():
        res.append(p.grad_sample.view(p.grad_sample.size(0), -1))
    return torch.cat(res, dim=1).squeeze()

# Clipt L2 norm of gradient to be at most C
def clip(grad, C):
    vec_norm = grad.norm(2, 1)
    multiplier = vec_norm.new(vec_norm.size()).fill_(1)
    multiplier[vec_norm.gt(C)] = C / vec_norm[vec_norm.gt(C)]
    grad *= multiplier.unsqueeze(1)
    return grad

# Privatizes the input gradients according to the privacy parameters
# in args
def privatize_grad(args, grad, mechanism=None):
    C = args.clip_val
    batch_size = grad.size(0)
    d = grad.size(1)
    grad_vec_np = np.zeros((grad.size(0),d+1))
    grad_vec_np[:,:-1] = grad.detach().cpu().numpy()
    # Complete norm to C then normalize
    v_m = np.min(C**2 - np.sum(grad_vec_np**2,1)+0.0001)
    if v_m > 0:
        grad_vec_np[:,-1] = np.sqrt(C**2 - np.sum(grad_vec_np**2,1)+0.0001)
    grad_vec_np = grad_vec_np/C
    epsilon = args.epsilon
    if mechanism == 'Gaussian' or mechanism == 'nonPrivate':
        grad_vec_np = None
        grad += torch.randn_like(grad).to(args.device) * args.clip_val * args.sigma
    else:
        W = None
        if mechanism == 'FastProjUnit-fixed':
            W = np.random.choice(a=[-1, 1], size=(grad_vec_np.shape[1]), p=[0.5, 0.5]) 
        elif mechanism == 'ProjUnit-fixed':
            W = np.random.normal(size=(args.k,grad_vec_np.shape[1]+1)) / k**0.5
        for i in range(grad_vec_np.shape[0]):
            grad_vec_np[i] = privatize_vector(grad_vec_np[i],epsilon,args.k,mechanism,args.p,args.gamma,args.sigma,True,W)
    if grad_vec_np is not None:
        grad = torch.from_numpy(grad_vec_np[:,:-1]).to(args.device).float()
    return C*grad.mean(0)

# updates the gradient of the model to be equal to the input gradient
def update_grad(model, grad):
    model.zero_grad()
    for p in model.parameters():
        size = p.data.view(1,-1).size(1)
        p.grad = grad[:size].view_as(p.data).clone()
        grad = grad[size:]
    return

def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    criterion = nn.CrossEntropyLoss()
    losses = []
    epsilon = args.epsilon
    for _batch_idx, (data, target) in enumerate(tqdm(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        grad = grad_to_vec(model)
        if args.mechanism != 'nonPrivate':
            grad = clip(grad, args.clip_val)
        grad_priv = privatize_grad(args, grad, args.mechanism)
        update_grad(model, grad_priv)
        optimizer.step()
        losses.append(loss.item())
        
    print(
        f"Train Epoch: {epoch} \t"
        f"Loss: {np.mean(losses):.6f} "
        f"(ε = {epsilon:.2f}, δ = {args.delta})"
    )


def test(model, device, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in tqdm(test_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(
                dim=1, keepdim=True
            )  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print(
        "\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n".format(
            test_loss,
            correct,
            len(test_loader.dataset),
            100.0 * correct / len(test_loader.dataset),
        )
    )
    return correct / len(test_loader.dataset)


def main():
    # Training settings
    parser = argparse.ArgumentParser(
        description="MNIST ProjUnit Experiments",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=600,
        metavar="B",
        help="Batch size",
    )
    parser.add_argument(
        "--test-batch-size",
        type=int,
        default=1024,
        metavar="TB",
        help="input batch size for testing",
    )
    parser.add_argument(
        "-n",
        "--epochs",
        type=int,
        default=10,
        metavar="N",
        help="number of epochs to train",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        metavar="LR",
        help="learning rate",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.5,
        help="Momentum",
    )
    parser.add_argument(
        "-c",
        "--clip-val",
        type=float,
        default=1.0,
        metavar="C",
        help="Clip per-sample gradients to this norm",
    )
    parser.add_argument(
        "--mechanism",
        type=str,
        default='Gaussian',
        choices=["nonPrivate","Gaussian", "PrivUnitG", "FastProjUnit", 
                 "ProjUnit", "FastProjUnit-fixed", "PrivHS", "RePrivHS", "SQKR"],
        help="Privacy mechanism",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=1000,
        help="number of projected dimensions for ProjUnit algorithms",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=1.0,
        metavar="S",
        help="Privacy parameter",
    )
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-5,
        metavar="D",
        help="Target delta",
    )
    parser.add_argument(
        "--num-rep",
        type=int,
        default=1,
        help="Number of Repetitions",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="GPU ID for this process",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=False,
        help="Save the trained model",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="../mnist",
        help="Where MNIST is/will be stored",
    )
    parser.add_argument(
        "--dir-res",
        type=str,
        default="MNIST_results",
        help="Directory for saving results",
    )
    args = parser.parse_args()
    device = torch.device(args.device)
    
    args.p = None
    args.sigma = None
    args.gamma = None
    if args.mechanism == 'nonPrivate':
        args.sigma = 0.0
    if args.mechanism == 'Gaussian':
    # set sigma according to epsilon
        if args.epsilon < 0: 
            args.sigma = 0.0
        else:
            args.sigma = calibrateAnalyticGaussianMechanism(args.epsilon, args.delta,args.clip_val)
    else:
        args.delta = 0
    if args.mechanism in ['PrivUnitG', 'FastProjUnit', 'ProjUnit', 'FastProjUnit-fixed']:
        # parameters for PrivG algorthms
        args.p = priv_unit_G_get_p(args.epsilon)
        args.gamma, args.sigma = get_gamma_sigma(args.p, args.epsilon)
    
    print("Mechanism is " + str(args.mechanism))
    
    output_file = "%s/mech_%s_num_rep_%d_epochs_%d_lr_%.2e_clip_%.2e_k_%d_epsilon_%.2e.pth" % (
        args.dir_res, args.mechanism, args.num_rep, args.epochs, args.lr, args.clip_val,
        args.k, args.epsilon
    )
    

    if os.path.exists(output_file):
        print('Existing result')
        return
        
        
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            args.data_root,
            train=True,
            download=True,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
                ]
            ),
        ),
        batch_size=args.batch_size,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(
            args.data_root,
            train=False,
            transform=transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Normalize((MNIST_MEAN,), (MNIST_STD,)),
                ]
            ),
        ),
        batch_size=args.test_batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )
    run_results = np.zeros((args.epochs,args.num_rep))
    
    for it in range(args.num_rep):
        model = GradSampleModule(ConvNet().to(device))
    
        num_parameters = sum([np.prod(layer.size()) for layer in model.parameters()])
        print("Num of parameters in model = %d" % num_parameters)
        
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    
    
        for epoch in range(1, args.epochs + 1):
            train(args, model, device, train_loader, optimizer, epoch)
            test_acc = test(model, device, test_loader)
            run_results[epoch-1][it] = test_acc
    
           
        np.save(output_file,run_results) #.detach().cpu().numpy())

if __name__ == "__main__":
   main()
    
    
    
