import torch
from torch.utils.data import DataLoader
from torch import nn
from NeuralNets import CustomVaexDataset, printNodes, sc, StraightNetwork
from matplotlib import pyplot as plt

from DataCore import snapshotPath

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

        if batch % 6 == 0:
            loss, current = loss.item(), batch * batch_size + len(X)
            print(f"loss: [{current:>5d}/{size:>5d}]  {loss:>7f}")




Ls = []
Cs = []


def snapshot(name=""):
    print("------------------")
    print("Autosave...")
    print("------------------")
    torch.save(model.state_dict(), snapshotPath)
    fig= plt.figure(figsize=(10,5))
    plt.subplot(121)
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
            correct += ((pred.argmax(0) - y).abs() < bmark).type(torch.float).sum().item()

    test_loss /= num_batches

    correct /= size * y.size()[1]
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    Ls.append(test_loss)
    Cs.append(correct)


p2 = "/Users/users/spirov/Blk/Nexus Project/Thesis-Project-Cosmic-Web/Data/Training/"
p3 = "/Users/users/spirov/Blk/Nexus Project/Thesis-Project-Cosmic-Web/Data/Testing/"


train_set = CustomVaexDataset(p2)
test_set = CustomVaexDataset(p3)


torch.set_default_dtype(torch.float64)

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




printNodes()


learning_rate = 1e-1 #sc/1e5
batch_size = 16
epochs = 1000
bmark = 1e3/sc
train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(test_set, batch_size=batch_size, shuffle=True)


#-------------------------------------------

loss_fn = nn.MSELoss() 

model = StraightNetwork()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) 


for t in range(epochs):
#    if t == 10:
#       optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate*10)
   
    if t%15==0 and t>0:
        snapshot(t)
    
    print(f"Epoch {t+1}\n-------------------------------")
    train_loop(train_dataloader, model, loss_fn, optimizer)
    test_loop(test_dataloader, model, loss_fn)
print("Done!")

snapshot()
