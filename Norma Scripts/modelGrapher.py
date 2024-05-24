import numpy as np
from matplotlib import pyplot as plt

import torch
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
from torch import nn

import vaex
import os

from cycler import cycler





plt.plot([1],[1])

font = {"weight": "normal", "size": 14}
plt.rcParams["axes.linewidth"] = 1.5  # set the value globally
plt.rc("font", **font)
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["figure.facecolor"] = "white"
_legend = {"frameon": True, "framealpha":0.7}
plt.rc("legend", **_legend)
plt.rcParams["axes.prop_cycle"] = cycler("color",('indigo','b','r','k','#ff7f0e','g'))
plt.show()



path = "/Users/users/spirov/Blk/Nexus Project/Thesis-Project-Cosmic-Web/"
p3 = "/Users/users/spirov/Blk/Nexus Project/Thesis-Project-Cosmic-Web/Data/Testing/"




lim = 5
sc = 1e4

class CustomVaexDataset(Dataset):
    def __init__(self,frameDir):
        self.lengt = len(os.listdir(frameDir))
        self.frameDir = frameDir
    
    def __len__(self):
        return self.lengt
        
        
    def __getitem__(self,idx):
        fil = self.frameDir+os.listdir(self.frameDir)[idx]
        df = vaex.open(fil)
        
        th = df.Th.values[::lim]
        fi = df.Fi.values[::lim]
        R = df.R.values[::lim]/sc
        CZ = df.CZ.values[::lim]/sc
        
        broken = np.array((CZ,th,fi))
        truth = np.array(R)
        
        return torch.tensor(broken), torch.tensor(truth)
                 
test_set = CustomVaexDataset(p3)
test_dataloader = DataLoader(test_set, batch_size=16, shuffle=True)
test_features, test_labels = next(iter(test_dataloader))


n = int((2**12)/4)
l=test_labels.size()[1]

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

    

torch.set_default_dtype(torch.float64)

model =StraightNetwork() #BottleneckNetwork() # StraightNetwork() #

path ="/Users/users/spirov/ThesisProject/modelSnapshot.pt"
model.load_state_dict(torch.load(path)) 


train_features, train_labels = next(iter(test_dataloader))
img = train_features[0].squeeze()
label = train_labels[0]

with torch.no_grad():
    mod = model(train_features)[0]
    


fig = plt.figure(figsize=(30,10))
plt.subplot(131,projection="polar")
plt.scatter(img[2],img[0],s=0.1*lim,alpha=0.7)
#plt.ylim(0,4e4)
plt.title("Broken")

plt.subplot(132,projection="polar")
plt.scatter(img[2],mod,s=0.1*lim,alpha=0.7)
plt.title("Model")

plt.subplot(133,projection="polar")
plt.scatter(img[2],label,s=0.1*lim,alpha=0.7)
#plt.ylim(0,4e4)
plt.title("Correct")

plt.suptitle("Data loaded from set")

fig.savefig('Predicted Model.png', dpi=fig.dpi)