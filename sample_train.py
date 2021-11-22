"Shamlessely adapted from https://docs.ray.io/en/latest/tune/examples/mnist_pytorch.html"
import ray
from ray import tune,init
from ray.tune.schedulers import AsyncHyperBandScheduler
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms
import os
import argparse
from filelock import FileLock

import numpy as np
import ctypes
class Net(torch.nn.Module):
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()
        self.hidden = torch.nn.Linear(n_feature, n_hidden)   # hidden layer
        self.predict = torch.nn.Linear(n_hidden, n_output)   # output layer

    def forward(self, x):
        x = F.relu(self.hidden(x))      # activation function for hidden layer
        x = self.predict(x)             # linear output
        return x


def train(config):

    x = torch.unsqueeze(torch.linspace(-1, 1, 100), dim=1)  # x data (tensor), shape=(100, 1)
    y = x.pow(2) + 0.2*torch.rand(x.size())  
    x,y=Variable(x),Variable(y)
    net=Net(1,config["hidden"],1)
    optimizer = torch.optim.Adam(net.parameters(), lr=config["lr"])
    loss_func = torch.nn.MSELoss()
    for i in range(config["steps"]):
        optimizer.zero_grad()
        yhat=net(x)
        loss=loss_func(y,yhat).mean()
        loss.backward()
        optimizer.step()

        tune.report(loss=loss.item())
    


    
        
        

if __name__ == "__main__":
    
    # for early stopping
    sched = AsyncHyperBandScheduler()
    # this connects to the workers spawned by the submit script
    config={"lr": tune.sample_from(lambda _: 10**(-np.random.randint(1,4 ))),
            "hidden": tune.sample_from(lambda _: np.random.randint(1,10 )),
            "steps": tune.sample_from(lambda _: np.random.randint(5,10 ))}
    init("auto")
    analysis = tune.run(
        train,
        metric="loss",
        mode="min",
        name="exp",
        scheduler=sched,
        num_samples=6,
        resources_per_trial={
            "cpu": 2,
            "gpu": 1  # set this for GPUs
        },
        
        config=config)
    print("Best config is:", analysis.best_config)