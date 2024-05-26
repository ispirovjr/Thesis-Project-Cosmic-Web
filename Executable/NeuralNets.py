from torch import nn
import torch
from torch.utils.data import Dataset

import vaex
import os
import numpy as np

from DataCore import DataSizeLimit

lim = 4  #datapoint reduction factor
sc = 1e4   #scaling factor to allow model to behave itself

l = int(DataSizeLimit/lim)

n = int((2**np.log2(l))/(2*lim))

class StraightNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(3 * l, n*4),
            nn.ReLU(),
            nn.Linear(4*n,n*4),
            nn.Sigmoid(),
            nn.Linear(4*n,4*n),
            nn.ReLU(),
            nn.Linear(n*4,4*n),
            nn.Sigmoid(),
            nn.Linear(n*4,n*4),
            nn.ReLU(),
            nn.Linear(4*n,l)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits




class BottleneckNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(3 * l, n*8),
            nn.ReLU(),
            nn.Linear(8*n,n*2),
            nn.Sigmoid(),
            nn.Linear(2*n,n),
            nn.ReLU(),
            nn.Linear(n,n),
            nn.Sigmoid(),
            nn.Linear(n,n*2),
            nn.ReLU(),
            nn.Linear(2*n,l)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class CustomVaexDataset(Dataset):
    def __init__(self, frameDir):
        self.lengt = len(os.listdir(frameDir))
        self.frameDir = frameDir

    def __len__(self):
        return self.lengt

    def __getitem__(self, idx):
        fil = self.frameDir + os.listdir(self.frameDir)[idx]
        df = vaex.open(fil)

        th = df.Th.values[::lim]
        fi = df.Fi.values[::lim]
        R = df.R.values[::lim] / sc
        CZ = df.CZ.values[::lim] / sc


        broken = np.array((CZ, th, fi))
        truth = np.array(R)

        return torch.tensor(broken), torch.tensor(truth)

def printNodes():
    print("----------------")
    print(f"{3 * l} -> {4 * n} -> {l}")
    print("----------------")