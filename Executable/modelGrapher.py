from matplotlib import pyplot as plt

import torch

from NeuralNets import CustomVaexDataset, StraightNetwork, BottleneckNetwork
from ModelMaker import snapshotPath

from cycler import cycler

from DataCore import L

font = {"weight": "normal", "size": 14}
plt.rcParams["axes.linewidth"] = 1.5  # set the value globally
plt.rc("font", **font)
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["figure.facecolor"] = "white"
_legend = {"frameon": True, "framealpha":0.7}
plt.rc("legend", **_legend)
plt.rcParams["axes.prop_cycle"] = cycler("color",('indigo','b','r','k','#ff7f0e','g'))


# Get the model we made
model =StraightNetwork() #BottleneckNetwork() # StraightNetwork() #
model.load_state_dict(torch.load(snapshotPath))



# Load a random dataset

dataPath = "/Users/users/spirov/Blk/Nexus Project/Thesis-Project-Cosmic-Web/Data/Testing/"

dataset = CustomVaexDataset(dataPath)

train_features, train_labels = dataset.__getitem__()
img = train_features[0].squeeze()
label = train_labels[0]

with torch.no_grad():
    mod = model(train_features)[0]
    

# Plot Data

plotscale = 1

fig = plt.figure(figsize=(10,5))
plt.subplot(131,projection="polar")
plt.scatter(img[2],img[0],s=0.1*plotscale,alpha=0.7)
plt.ylim(0,L/2)
plt.title("Broken")

plt.subplot(132,projection="polar")
plt.scatter(img[2],mod,s=0.1*plotscale,alpha=0.7)
plt.title("Model")
plt.ylim(min(mod),L/2)

plt.subplot(133,projection="polar")
plt.scatter(img[2],label,s=0.1*plotscale,alpha=0.7)
plt.title("Correct")
plt.ylim(0,L/2)

plt.suptitle("Data loaded from set")
fig.savefig('./Model Figures/Predicted Model.png', dpi=fig.dpi)