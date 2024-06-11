import torch
from torch.utils.data import DataLoader
from torch import nn
from NeuralNets import CustomVaexDataset, printNodes, sc, StraightNetwork, Unet, Stefann
from matplotlib import pyplot as plt

from DataCore import snapshotPath



learning_rate = 1e-1#sc/1e5
batch_size = 16
epochs = 10000
bmark = 1e2/sc




device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print("----------------------")
print(f"Using {device} device")
print("----------------------")

if device == "cuda":
    cu = torch.device(device)
    torch.set_default_device(cu)

    


def train_loop(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    # Set the model to training mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 8 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: [{current:>5d}/{size:>5d}]  {loss:>7f}")




Ls = []
Cs = []

plotscale = 1

def snapshot(name=""):
    print("------------------")
    print("Autosave...")
    print("------------------")
    torch.save(model.state_dict(), snapshotPath)
    
    fig= plt.figure(figsize=(10,5))
    plt.subplot(121)
    plt.yscale('log')
    plt.plot(Ls, marker="+")
    plt.title("Loss")
    plt.subplot(122)
    plt.plot(Cs, marker="v")
    plt.title("Correctness")
    plt.suptitle(f"Snapshot {name}")
    fig.savefig('./Model Figures/Model State.png', dpi=fig.dpi)

        


def test_loop(dataloader, model, loss_fn):
    # Set the model to evaluation mode - important for batch normalization and dropout layers
    # Unnecessary in this situation but added for best practices
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    # Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
    # also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
    with torch.no_grad():
        for X, y in dataloader:
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += ((pred - y).abs() < bmark).type(torch.int).sum().item()
            
    test_loss /= num_batches

    correct /= y.size()[0] * y.size()[1]
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    Ls.append(test_loss)
    Cs.append(correct)



    
p2 = "/Users/users/spirov/Blk/Nexus Project/Thesis-Project-Cosmic-Web/Data/Training/"
p3 = "/Users/users/spirov/Blk/Nexus Project/Thesis-Project-Cosmic-Web/Data/Testing/"


train_set = CustomVaexDataset(p2)
test_set = CustomVaexDataset(p3)

torch.set_default_dtype(torch.float64)


img, label = test_set.__getitem__(0)
imj = img.cpu()
labe=label.cpu()

def makePretty(n,model,name="CurrentAttempt"):
    with torch.no_grad():
        mod = model(img.reshape(1,3,len(label))).cpu()

    
    fig = plt.figure(figsize=(8,4))
    
    plt.subplot(131,projection="polar")
    plt.ylim(0,1.75)
    plt.scatter(imj[2],imj[0],s=0.1*plotscale,alpha=0.7)
    plt.title("Distorted")

    plt.subplot(132,projection="polar")
    plt.ylim(0,1.75)
    plt.scatter(imj[2],mod,s=0.1*plotscale,alpha=0.7)
    plt.title("AI Model")

    plt.subplot(133,projection="polar")
    plt.ylim(0,1.75)
    plt.scatter(imj[2],labe,s=0.1*plotscale,alpha=0.7)
    plt.title("Physical")

    plt.suptitle(f"State at {n}")

    fig.savefig(f'./Model Figures/Anims/anim{name}-{n}.png', dpi=fig.dpi)





printNodes()

gen = torch.Generator(device='cuda')

train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True,generator = gen)
test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=True,generator = gen)


#-------------------------------------------

#loss_fn = nn.MSELoss() 
loss_fn = Stefann(1)  #HuberLoss, custom


model = Unet()

#model.load_state_dict(torch.load("/Users/users/spirov/ThesisProject/Snapshots/StefannSGD.pt"))


optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) 

#scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99)
#scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3500,7000], gamma=3e-1)

for t in range(epochs):

    if t%10 ==0:
        makePretty(t,model,"Stefann 3.75 %8 SDG->Adam lower lr ")
    
    if t%100==0 and t>0:
        snapshot(t)
    
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)

    if Cs[-1]>.2:
        learning_rate /= 50
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate) 
        
    #scheduler.step()


print("Done!")

snapshot()
