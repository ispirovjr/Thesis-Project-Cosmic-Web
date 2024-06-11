from matplotlib import pyplot as plt

import torch

from NeuralNets import CustomVaexDataset, StraightNetwork, Unet, sc

from cycler import cycler

from DataCore import L, snapshotPath

font = {"weight": "normal", "size": 14}
plt.rcParams["axes.linewidth"] = 1.5  # set the value globally
plt.rc("font", **font)
plt.rcParams["axes.facecolor"] = "white"
plt.rcParams["figure.facecolor"] = "white"
_legend = {"frameon": True, "framealpha":0.7}
plt.rc("legend", **_legend)
plt.rcParams["axes.prop_cycle"] = cycler("color",('indigo','b','r','k','#ff7f0e','g'))


torch.set_default_dtype(torch.float64)
# Get the model we made
model =Unet() #BottleneckNetwork() # StraightNetwork() #

GoodPath = "/Users/users/spirov/ThesisProject/Snapshots/NewGoodSnap.pt"

model.load_state_dict(torch.load(snapshotPath,map_location=torch.device('cpu')))



# Load a random dataset

dataPath = "/Users/users/spirov/Blk/Nexus Project/Thesis-Project-Cosmic-Web/Data/Testing/"

dataset = CustomVaexDataset(dataPath)

img, label = dataset.__getitem__(0)


with torch.no_grad():
    mod = model(img.reshape(1,3,len(label)))
    

# Plot Data

plotscale = 1

ran = L/2
ran /= sc

fig = plt.figure(figsize=(10,5))
plt.subplot(131,projection="polar")
plt.scatter(img[2],img[0],s=0.1*plotscale,alpha=0.7)
#plt.ylim(0,ran)
plt.title("Distorted")

plt.subplot(132,projection="polar")
plt.scatter(img[2],mod,s=0.1*plotscale,alpha=0.7)
plt.title("AI Model")
#plt.ylim(min([mod.min(),0]),ran)

plt.subplot(133,projection="polar")
plt.scatter(img[2],label,s=0.1*plotscale,alpha=0.7)
plt.title("Physical")
#plt.ylim(0,ran)

plt.suptitle("Data loaded from set")
fig.savefig('./Model Figures/Predicted Model.png', dpi=fig.dpi)