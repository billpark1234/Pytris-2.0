import os, os.path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import DataLoader, random_split
from extractor import Extractor
from agents import CNN, CNN2
import numpy as np


"""
Pipeline
Stage 1: Extractor extracts instance and label from raw board pictures.
This stage 1 instance is 22 by 10 2D array and an integer in range (1..7) inclusive.
The stage 1 label is two integers where first one is in range (-2..9) inclusive and second one is in range (0..3) inclusive.
They represent position and rotation respectively.


Stage 2: The arrays in the instances are converted into binary arrays. The position variable in the label is rescaled to (0..11),
and it is combined with rotation variable to produce a new label c = pos * 4 + rot.


Stage 3: The input is fed into a convolutional neural network with an embedding for tetromino type.
This is a multiclass classification task using softmax and cross-entropy loss.
The input to this NN is 2D binary array of shape (22, 10) and an integer from [1..7]
The output to this NN is a list of logits for 48 categories for c.

Stage 4: In game engine, this NN agent gives those logits. We use softmax to find the value c
with the highest score. This c is decoded into position and rotation.
The position is rescaled back to (-2..9) by subtracting 2.
"""

class TetrisDataset(Dataset):
    def __init__(self, dir, binaryBoard=True):

        """
        dir is the directory with all the json files that represent replays.
        """
        self.extractor = Extractor()
        self.dir = dir
        self.binaryBoard = binaryBoard
        
        self._instances = []
        self._labels = []
        n = len([name for name in os.listdir(self.dir) if os.path.isfile(os.path.join(self.dir, name))])
        
        for i in range(n):
            filename = os.path.join(self.dir, str(i+1) + ".json")
            self.extractor.load(filename)
            
            #Stage 1
            instance, labels = self.extractor.extractExamples()
            
            self._instances += instance
            self._labels += labels
                
    
    def __len__(self):
        return len(self._instances)
    
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
                idx = idx.tolist()
        
        #Stage 2
        
        
        board = self._instances[idx][0]
        piece = self._instances[idx][1]
        
        if self.binaryBoard:
            board = self._convertToBinary(board)
        
        ##IMPORTANT: tensor should be one dim higher than the argument because dataloader will add batch_size
        board = torch.tensor(board, dtype=torch.float32).unsqueeze(0)
        piece = torch.tensor(piece, dtype=torch.long).unsqueeze(0)
    
        pos, rot = self._labels[idx]
        c = (pos + 2) * 4 + rot
        c = torch.tensor(c, dtype=torch.long)

        return board, piece, c
    
    
    def _convertToBinary(self, board):
        board = board.copy()
        board[board != 0] = 1
        return board


net = CNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
numEpochs = 50
batch_size = 3


print("Loading the dataset..")
dataset = TetrisDataset("boards", True)
train_size = int(0.6 * len(dataset))
validation_size = int(0.3 * len(dataset))
test_size = len(dataset) - train_size - validation_size
trainset, validationset, testset = random_split(dataset, [train_size, validation_size, test_size])

trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
validationLoader = DataLoader(validationset, batch_size=batch_size, shuffle=True)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)
print("Finished loading the dataset.")
print(f"Trainset size: {len(trainset)}")
print(f"Validation set size: {len(validationset)}")
print(f"Test set size: {len(testset)}")


train_loss = []
validation_loss = []

for epoch in range(numEpochs):  # loop over the dataset multiple times
    running_loss = []
    net.train()
    for i, data in enumerate(trainloader, 0):
        board, piece, labels = data
        optimizer.zero_grad()
        outputs = net(board, piece)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss.append(loss.item())

    avg_loss = np.array(running_loss).mean()
    train_loss.append(avg_loss)

    validation_running_loss = []
    net.eval()
    for i, data in enumerate(validationLoader, 0):
        board, piece, labels = data
        outputs = net(board, piece)
        loss = criterion(outputs, labels)
        validation_running_loss.append(loss.item())
    validation_loss.append(np.array(validation_running_loss).mean())

    print(f'Epoch {epoch + 1}, train: {train_loss[-1]:.3f}, validation: {validation_loss[-1]:.3f}')

print('Finished Training')

# Plot the running loss
import matplotlib.pyplot as plt

plt.plot(range(1, numEpochs + 1), train_loss, label='train loss')
plt.plot(range(1, numEpochs + 1), validation_loss, label='validation loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Running Loss vs. Epochs')
plt.legend()
plt.show()



## Save and evaluate performance on the test set
PATH = 'model2.pth'
torch.save(net.state_dict(), PATH)
net.load_state_dict(torch.load(PATH, weights_only=True))

correct = 0
total = 0
with torch.no_grad():
    for data in testloader:
        board, piece, labels = data
        outputs = net(board, piece)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print(f'Accuracy of the network on the test set of size: {len(testloader)}. {100 * correct // total} %')