import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader

import torch.optim as optim

import loss_for_evidential

def init_kaiming_normal(layer):
    if (type(layer) == nn.Linear) or (type(layer) == nn.Conv1d):
        torch.nn.init.xavier_uniform_(layer.weight)

# https://colab.research.google.com/drive/14YIibn8mcEehlUcKxVhK3BYSa10loLzZ?usp=sharing
class NewResNet(nn.Module):
    def __init__(self, unc_control='no'):
        super(NewResNet, self).__init__()

        self.conv1 = nn.Conv1d(4, 196, kernel_size=19, padding=19//2)
        self.bn1 = nn.BatchNorm1d(196)
        self.silu = nn.SiLU()
        self.dropout1 = nn.Dropout(0.2)
        self.resid_block1 =  self._residual_block(196, kernel_size=3, num_layers=5, dropout=0.1)

        self.maxpool1 = nn.MaxPool1d(5)
        self.maxpool2 = nn.MaxPool1d(5)
        self.avgpool = nn.AvgPool1d(9)

        self.conv2 = nn.Conv1d(196, 256, kernel_size=7, padding=7//2)
        self.bn2 = nn.BatchNorm1d(256)
        self.dropout2 = nn.Dropout(0.2)

        self.fc1 = nn.LazyLinear(256)
        self.bn3 = nn.BatchNorm1d(256)
        self.dropout5 = nn.Dropout(0.5)

        self.flatten = nn.Flatten()

        self.fc2 = nn.LazyLinear(256)
        self.bn4 = nn.BatchNorm1d(256)

        self.unc_control=unc_control
        if unc_control=='no':
            #self.output_dim=1
            self.output = nn.Linear(256, 1)
        elif unc_control=='heteroscedastic':
            #self.output_dim=2
            self.output = nn.Linear(256, 2)
        elif unc_control=='evidential':
            self.output = loss_for_evidential.DenseNormalGamma(256, 1)
        
        # Initialize weights according to He initialization
        self.apply(init_kaiming_normal)


    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.silu(x)
        x = self.dropout1(x)
        resid_block1 = self.resid_block1(x)
        x = x.clone() +  resid_block1
        x = self.silu(x)
        x = self.dropout2(x)
        x = self.maxpool1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.silu(x)
        #x = self.dropout2(x)
        #resid_block2 = self.resid_block2(x)
        #x = x.clone() +  resid_block2
        #x = self.relu(x)
        x = self.dropout2(x)
        x = self.maxpool2(x)

        x = self.fc1(x)
        x = self.bn3(x)
        x = self.silu(x)
        x = self.dropout5(x)

        x = self.avgpool(x)
        x = self.flatten(x)

        x = self.fc2(x)
        x = self.bn4(x)
        x = self.silu(x)
        x = self.dropout5(x)

        if self.unc_control=='no' or self.unc_control=='heteroscedastic':
            output = self.output(x)
            return output
        elif self.unc_control=='evidential':
            mu, logv, alpha, beta = self.output(x)
            return mu, logv, alpha, beta       


    def _residual_block(self, input_dim, kernel_size=3, num_layers=5, dropout=0.1):

        layers = [nn.Conv1d(input_dim, input_dim, kernel_size=kernel_size, padding=kernel_size//2, bias=False, dilation=1)]
        layers.append(nn.BatchNorm1d(input_dim))

        base_rate = 2
        for i in range(1,num_layers):
            layers.append(nn.SiLU())
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Conv1d(input_dim, input_dim, kernel_size=kernel_size, padding=(kernel_size//2 + 1)**i, bias=False, dilation=base_rate**i))
            layers.append(nn.BatchNorm1d(input_dim))
        return nn.Sequential(*layers)
