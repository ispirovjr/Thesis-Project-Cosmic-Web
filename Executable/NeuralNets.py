from torch import nn
import torch
from torch.utils.data import Dataset

import vaex
import os
import numpy as np
from torch.nn.modules.loss import _Loss

from DataCore import DataSizeLimit

lim = 8  #datapoint reduction factor
sc = 3.75e4   #scaling factor to allow model to behave itself

l = int(DataSizeLimit/lim)

n = int(2**int(np.log2(l)))

class StraightNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(3 * l, n),
            nn.ReLU(),
            nn.Linear(n,n),
            nn.Dropout(0.35),
            nn.ReLU(),
            nn.Linear(n,n),
            nn.ReLU(),
            nn.Linear(n,n),
            nn.ReLU(),
            nn.Dropout(0.35),
            nn.Linear(n,n),
            nn.ReLU(),
            nn.Linear(n,l),
            nn.ReLU()
        )


    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits



f = 16
class Unet(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(3 * l, int(n/(2*f))),
            nn.ReLU(),
            #nn.Dropout(0.35),
            nn.Linear(int(n/(2*f)),int(n/(4*f))),
            nn.Sigmoid(),
            nn.Linear(int(n/(4*f)),int(n/(8*f))),
            nn.ReLU(),
            nn.Linear(int(n/(8*f)),int(n/(2*f))),
            nn.ReLU(),
            #nn.Dropout(0.35),
            nn.Linear(int(n/(2*f)),int(n/f)),
            nn.ReLU(),
            nn.Linear(int(n/f),l)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


class Stefann(_Loss):  # STandard dEviation Framework for Astrophysical Neural Networks
    def __init__(self, weight=3):
        super().__init__()
        self.mult = weight

    def forward(self, output, target):
        loss1 = torch.mean((output - target) ** 2)
        tstd = target.std().item()
        ostd = output.std().item() + 1e-5

        a = tstd / ostd
        b = ostd / tstd

        loss2 = (a + b) / 2 - 1

        loss = loss1 + loss2 * self.mult
        # print(loss2)

        return loss


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
    print(f"{3 * l} -> { n} -> {l}")
    print("----------------")