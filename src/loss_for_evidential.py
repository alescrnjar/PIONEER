import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# https://github.com/aamini/evidential-deep-learning/blob/main/hello_world.py

# Define the DenseNormalGamma layer
class DenseNormalGamma(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DenseNormalGamma, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim * 4)

    def forward(self, x):
        out = self.fc(x)
        mu, logv, alpha, beta = torch.chunk(out, 4, dim=-1)
        return mu, logv, alpha, beta

# Define the PyTorch model
class EvidentialModel(nn.Module):
    def __init__(self):
        super(EvidentialModel, self).__init__()
        self.fc1 = nn.Linear(64, 64)
        self.fc2 = nn.Linear(64, 64)
        self.output_layer = DenseNormalGamma(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        mu, logv, alpha, beta = self.output_layer(x)
        return mu, logv, alpha, beta

# Evidential loss function
def evidential_loss(y, mu, v, alpha, beta, lambda_):
    # Assuming y is the ground truth and the output is (mu, logv, alpha, beta)
    twoBlambda = 2 * beta * (1 + lambda_)
    nll = 0.5 * torch.log(np.pi / v) \
          - alpha * torch.log(twoBlambda) \
          + (alpha + 0.5) * torch.log(v * (y - mu)**2 + twoBlambda) \
          + torch.lgamma(alpha) \
          - torch.lgamma(alpha + 0.5)
    return nll.mean()

