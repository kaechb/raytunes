"Shamlessely adapted from https://docs.ray.io/en/latest/tune/examples/mnist_pytorch.html"
import os
import argparse
from filelock import FileLock
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms

import ray
from ray import tune,init
from ray.tune.schedulers import AsyncHyperBandScheduler
import ctypes
libgcc_s = ctypes.CDLL("libgcc_s.so.1")
# Change these values if you want the training to run quicker or slower.
EPOCH_SIZE = 512
TEST_SIZE = 256


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 3, kernel_size=3)
        self.fc = nn.Linear(192, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 3))
        x = x.view(-1, 192)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)

def train(model, optimizer, train_loader, device=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if batch_idx * len(data) > EPOCH_SIZE:
            return
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()


def test(model, data_loader, device=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(data_loader):
            if batch_idx * len(data) > TEST_SIZE:
                break
            data, target = data.to(device), target.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            total += target.size(0)
            correct += (predicted == target).sum().item()

    return correct / total


def get_data_loaders():
    mnist_transforms = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.1307, ), (0.3081, ))])

    # We add FileLock here because multiple workers will want to
    # download data, and this may cause overwrites since
    # DataLoader is not threadsafe.
    with FileLock(os.path.expanduser("~/raytunes/data.lock")):
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                "~/data",
                train=True,
                download=True,
                transform=mnist_transforms),
            batch_size=64,
            shuffle=True)
        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST(
                "~/data",
                train=False,
                download=True,
                transform=mnist_transforms),
            batch_size=64,
            shuffle=True)
    return train_loader, test_loader


def train_mnist(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, test_loader = get_data_loaders()
    model = ConvNet().to(device)

    optimizer = optim.SGD(
        model.parameters(), lr=config["lr"], momentum=config["momentum"])

    while True:
        train(model, optimizer, train_loader, device)
        acc = test(model, test_loader, device)
        # Set this to run Tune.
        tune.report(mean_accuracy=acc)


if __name__ == "__main__":
    
    # for early stopping
    #sched = AsyncHyperBandScheduler()
    # this connects to the workers spawned by the submit script
    init("auto")
    analysis = tune.run(
        train_mnist,
        metric="mean_accuracy",
        mode="max",
        name="exp",
        #scheduler=sched,
        stop={
            "mean_accuracy": 0.98,
            "training_iteration": 100
        },
        resources_per_trial={
            "cpu": 2,
            "gpu": 1  # set this for GPUs
        },
        num_samples=1,
        config={
            "lr": tune.loguniform(1e-4, 1e-2),
            "momentum": tune.uniform(0.1, 0.9),
        })

    print("Best config is:", analysis.best_config)