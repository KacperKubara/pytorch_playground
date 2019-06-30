import numpy as np
import torch
import helper
import matplotlib.pyplot as plt
from pytorchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor(),
transforms.Normalize((0.5,), (0.5,)), ])

trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

dataiter = iter(trainloader)
images, labels = dataiter.next()

# Build a NN - 784 -> 256 -> 10
def activation(x):
    return 1/(1+torch.exp(-x))

inputs = images.view(images.shape[0], -1) 
n_hidden = 256
n_outputs = 10

w1 = torch.randn(784, n_hidden)
w2 = torch.randn(n_hidden, n_outputs)

b1 = torch.randn(784, n_hidden)
b2 = torch.randn(n_hidden, n_outputs)

h = torch.mm(inputs, w1) + b1
out = torch.mm(h, w2)  + b2