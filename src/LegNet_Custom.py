# https://github.com/autosome-ru/human_legnet/blob/main/model.py
import pytorch_lightning as pl
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class DataModule(pl.LightningDataModule):
    def __init__(self, x_train, y_train, x_valid, y_valid, x_test, y_test, batch_size=128):
        super().__init__()
        self.batch_size = batch_size
        self.x_train = torch.from_numpy(x_train)
        self.y_train = torch.from_numpy(y_train)
        self.x_valid = torch.from_numpy(x_valid)
        self.y_valid = torch.from_numpy(y_valid)
        self.x_test = torch.from_numpy(x_test)
        self.y_test = torch.from_numpy(y_test)

    def train_dataloader(self):
        train_dataset = TensorDataset(self.x_train, self.y_train) # tensors are index-matched
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True) # sets of (x, x', y) will be shuffled

    def val_dataloader(self):
        valid_dataset = TensorDataset(self.x_valid, self.y_valid)
        return DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)

    def test_dataloader(self):
        test_dataset = TensorDataset(self.x_test, self.y_test)
        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

def initialize_weights(m):
    if isinstance(m, nn.Conv1d):
        n = m.kernel_size[0] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2 / n))
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.constant_(m.weight.data, 1)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.001)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

class SELayer(nn.Module):
    def __init__(self, inp, reduction=4):
        super(SELayer, self).__init__()
        self.fc = nn.Sequential(
                nn.Linear(inp, int(inp // reduction)),
                nn.SiLU(),
                #nn.Dropout(0.1), #AC
                nn.Linear(int(inp // reduction), inp),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, = x.size()
        y = x.view(b, c, -1).mean(dim=2)
        y = self.fc(y).view(b, c, 1)
        return x * y

class EffBlock(nn.Module):
    def __init__(self, in_ch, ks, resize_factor, activation, out_ch=None, se_reduction=None):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = self.in_ch if out_ch is None else out_ch
        self.resize_factor = resize_factor
        self.se_reduction = resize_factor if se_reduction is None else se_reduction
        self.ks = ks
        self.inner_dim = self.in_ch * self.resize_factor

        block = nn.Sequential(
                        nn.Conv1d(
                            in_channels=self.in_ch,
                            out_channels=self.inner_dim,
                            kernel_size=1,
                            padding='same',
                            bias=False
                       ),
                       nn.BatchNorm1d(self.inner_dim),
                       activation(),
                       #nn.Dropout(0.1), #AC

                       nn.Conv1d(
                            in_channels=self.inner_dim,
                            out_channels=self.inner_dim,
                            kernel_size=ks,
                            groups=self.inner_dim,
                            padding='same',
                            bias=False
                       ),
                       nn.BatchNorm1d(self.inner_dim),
                       activation(),
                       #nn.Dropout(0.1), #AC,
                       SELayer(self.inner_dim, reduction=self.se_reduction),
                       nn.Conv1d(
                            in_channels=self.inner_dim,
                            out_channels=self.in_ch,
                            kernel_size=1,
                            padding='same',
                            bias=False
                       ),
                       nn.BatchNorm1d(self.in_ch),
                       activation(),
                       #nn.Dropout(0.1) #AC
        )

        self.block = block

    def forward(self, x):
        return self.block(x)

class LocalBlock(nn.Module):
    def __init__(self, in_ch, ks, activation, out_ch=None):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = self.in_ch if out_ch is None else out_ch
        self.ks = ks

        self.block = nn.Sequential(
                       nn.Conv1d(
                            in_channels=self.in_ch,
                            out_channels=self.out_ch,
                            kernel_size=self.ks,
                            padding='same',
                            bias=False
                       ),
                       nn.BatchNorm1d(self.out_ch),
                       activation(),
                       #nn.Dropout(0.1) #AC
        )

    def forward(self, x):
        return self.block(x)

class ResidualConcat(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return torch.concat([self.fn(x, **kwargs), x], dim=1)

class MapperBlock(nn.Module):
    def __init__(self, in_features, out_features, activation=nn.SiLU):
        super().__init__()
        self.block = nn.Sequential(
            nn.BatchNorm1d(in_features),
            nn.Conv1d(in_channels=in_features,
                      out_channels=out_features,
                      kernel_size=1),
            nn.Dropout(0.1) #AC
        )

    def forward(self, x):
        return self.block(x)

class LegNet_Custom(nn.Module):
    def __init__(self,
                 
                 in_ch,
                 stem_ch=64, #https://github.com/autosome-ru/human_legnet/blob/main/core.py
                 stem_ks=11, #0.81:11, #Orig Paper: 11 #PK: 4
                 ef_ks=9, #0.81:9,
                 ef_block_sizes=[80, 96, 112, 128],
                 pool_sizes=[2, 2, 2, 2],
                 resize_factor=4,
                 activation=nn.SiLU,
                 unc_control='no',
                 ):

        super().__init__()
        assert len(pool_sizes) == len(ef_block_sizes)

        self.in_ch = in_ch
        self.stem = LocalBlock(in_ch=in_ch,
                               out_ch=stem_ch,
                               ks=stem_ks,
                               activation=activation)

        blocks = []

        in_ch = stem_ch
        out_ch = stem_ch
        for pool_sz, out_ch in zip(pool_sizes, ef_block_sizes):
            blc = nn.Sequential(
                ResidualConcat(
                    EffBlock(
                        in_ch=in_ch,
                        out_ch=in_ch,
                        ks=ef_ks,
                        resize_factor=resize_factor,
                        activation=activation)
                ),
                LocalBlock(in_ch=in_ch * 2,
                           out_ch=out_ch,
                           ks=ef_ks,
                           activation=activation),
                nn.MaxPool1d(pool_sz) if pool_sz != 1 else nn.Identity(),
                nn.Dropout(0.1), #AC
            )
            in_ch = out_ch
            blocks.append(blc)
        self.main = nn.Sequential(*blocks)

        self.mapper = MapperBlock(in_features=out_ch,
                                  out_features=out_ch * 2)
        if unc_control=='no':
            self.head = nn.Sequential(nn.Linear(out_ch * 2, out_ch * 2),
                                      nn.Dropout(0.5), #AC
                                   nn.BatchNorm1d(out_ch * 2),
                                   activation(),
                                   nn.Linear(out_ch * 2, 1))

    def forward(self, x):
        x = self.stem(x)
        x = self.main(x)
        x = self.mapper(x)
        x =  F.adaptive_avg_pool1d(x, 1)
        x = x.squeeze(-1)
        x = self.head(x)
        return x

