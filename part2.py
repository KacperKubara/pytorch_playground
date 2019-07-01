from pytorchvision import datasets, transforms
import torch
import numpy as np
import matplotlib.pyplot as plt
import helper

transform = transforms.Compose([transforms.ToTensor(),
transforms.Normalize((0.5,), (0.5,)), ])

trainset = datasets.MNIST('~/.pytorch/MNIST_data/', download=True, train=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)

dataiter = iter(trainloader)
images, labels = dataiter.next()

# Build a NN - 784 -> 256 -> 10
def activation(x):
    return 1/(1+torch.exp(-x))

# Reshape the image to the image size
inputs = images.view(images.shape[0], -1) 
n_hidden = 256
n_outputs = 10

# Define the weights for the input and hidden layer
w1 = torch.randn(784, n_hidden)
w2 = torch.randn(n_hidden, n_outputs)

# Define the bias for the input and hidden layer
b1 = torch.randn(784, n_hidden)
b2 = torch.randn(n_hidden, n_outputs)

# Connect two layers
h = torch.mm(inputs, w1) + b1
out = torch.mm(h, w2)  + b2

def softmax(x):
    ## TODO: Implement the softmax function here
    return torch.exp(x)/torch.sum(torch.exp(x), dim=1).view(-1, 1)
# Here, out should be the output of the network in the previous excercise with shape (64,10)
probabilities = softmax(out)
