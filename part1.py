import torch
# To convert data from pytorch to numpy use either:
# from_numpy()
# numpy()
def activation(x):
    return 1/(1+torch.exp(-x))
def neuron():
    features = torch.randn((1, 5))
    weights = torch.randn_like(features)
    bias = torch.randn((1, 1))
    node = torch.sum(torch.mm(features, weights.view(5, 1)), bias)
    output = activation(node)

def neuron_network():
    torch.manual_seed(7)
    features = torch.randn((1, 3))
    n_input = features.shape[1]
    n_hidden = 2
    n_output = 1

    W1 = torch.randn(n_input, n_hidden)
    W2 = torch.randn(n_hidden, n_output)

    B1 = torch.randn((1, n_hidden))
    B2 = torch.randn((1, n_output))

    H = activation(torch.mm(features, W1) + B1)
    output = activation(torch.mm(H, W2) + B2)
    return output


if __name__=='__main__':
    print(neuron_network())