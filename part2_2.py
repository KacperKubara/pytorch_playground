import pytorch.nn.functional as F
from torch import nn

class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden_0 = nn.Linear(784, 128)
        self.hidden_1 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 10)
    def forward(self, x):
        x = F.relu(self.hidden_0(x))
        x = F.relu(self.hidden_1(x))
        x = F.softmax(self.output(x), dim=1)
        
        return x