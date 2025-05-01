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
The input to this NN is 2D binary array of shape (20, 10) and an integer from [1..7]
The output to this NN is a list of logits for 48 categories for c.

Stage 4: In game engine, this NN agent gives those logits. We use softmax to find the value c
with the highest score. This c is decoded into position and rotation.
The position is rescaled back to (-2..9) by subtracting 2.
"""

class TetrisDataset(Dataset):
    def __init__(self, dir, binaryBoard=True):

        """
        Tetris Dataset class for loading and processing Tetris replay data.
        
        Instance space: (board, piece) where board is a 2D array of shape (20, 10) and piece is an integer in range (1..7) inclusive.
        Label space: c, an integer in range (0..47) inclusive. This is a combination of position and rotation.
        (The position is in range (-2..9) inclusive and the rotation is in range (0..3) inclusive)

        
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
        c = torch.tensor(c, dtype=torch.long) #combine position and rotation

        return board, piece, c
    
    
    def _convertToBinary(self, board):
        board = board.copy()
        board[board != 0] = 1
        return board


class EarlyStopper:
    def __init__(self, patience=1, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.min_validation_loss = float('inf')

    def early_stop(self, validation_loss):
        if validation_loss < self.min_validation_loss:
            self.min_validation_loss = validation_loss
            self.counter = 0
        elif validation_loss > (self.min_validation_loss + self.min_delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False


class Trainer:
    def __init__(self, model, loss_fn, optimizer, dataset, early_stopper=None):
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.dataset = dataset
        self.train_loss = []
        self.validation_loss = []
        self.early_stopper = early_stopper

    
    def to(self, device):
        self.device = device
        self.model.to(device)

    def load_dataset(self, dataset, data_division):
        print("Loading the dataset..")
        train_size = int(data_division[0] * len(self.dataset))
        validation_size = int(data_division[1] * len(self.dataset))
        test_size = len(self.dataset) - train_size - validation_size
        trainset, validationset, testset = random_split(self.dataset, [train_size, validation_size, test_size])

        trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
        validationLoader = DataLoader(validationset, batch_size=batch_size, shuffle=True)
        testloader = DataLoader(testset, batch_size=batch_size, shuffle=True)

        print("Finished loading the dataset.")
        print(f"Trainset size: {len(trainset)}")
        print(f"Validation set size: {len(validationset)}")
        print(f"Test set size: {len(testset)}")

        self.trainloader = trainloader
        self.validationLoader = validationLoader
        self.testloader = testloader


    def train(self, num_epochs, batch_size):
        trainloader = self.trainloader
        validationLoader = self.validationLoader
        criterion = self.loss_fn
        optimizer = self.optimizer
        stopper = self.early_stopper

        for epoch in range(numEpochs):  # loop over the dataset multiple times
            running_loss = []
            self.model.train()
            for i, data in enumerate(trainloader, 0):
                board, piece, labels = data

                board = board.to(self.device)
                piece = piece.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = self.model(board, piece)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running_loss.append(loss.item())

            avg_loss = np.array(running_loss).mean()
            self.train_loss.append(avg_loss)

            validation_running_loss = []
            self.model.eval()
            for i, data in enumerate(validationLoader, 0):
                board, piece, labels = data
                board = board.to(self.device)
                piece = piece.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(board, piece)
                loss = criterion(outputs, labels)
                validation_running_loss.append(loss.item())
                avg_validation_loss = np.array(validation_running_loss).mean()
            self.validation_loss.append(avg_validation_loss)

            if stopper is not None and stopper.early_stop(avg_validation_loss):
                print(f"Early stopping at epoch {epoch + 1}")
                break

            print(f'Epoch {epoch + 1}, train: {self.train_loss[-1]:.3f}, validation: {self.validation_loss[-1]:.3f}')

        print('Finished Training')


    def save_model(self, PATH):
        torch.save(self.model.state_dict(), PATH)
        self.model.load_state_dict(torch.load(PATH, weights_only=True))


    def plot(self):
        import matplotlib.pyplot as plt
        plt.plot(range(1, len(self.train_loss) + 1), self.train_loss, label='train loss')
        plt.plot(range(1, len(self.validation_loss) + 1), self.validation_loss, label='validation loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Running Loss vs. Epochs')
        plt.legend()
        plt.show()


    def evaluate(self):
        testloader = self.testloader
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                board, piece, labels = data
                board = board.to(self.device)
                piece = piece.to(self.device)
                labels = labels.to(self.device)
                outputs = self.model(board, piece)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        print(f'Accuracy of the network on the test set of size: {len(testloader)}. {100 * correct // total} %')



from agents import VisionTransformer
num_layers = 8
emb_size = 32
num_head = 4
num_class= 48
patch_size=2

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = VisionTransformer(num_layers=num_layers, img_size=20, emb_size=emb_size, patch_size=patch_size, num_head=num_head, num_class=num_class)
criterion = nn.CrossEntropyLoss()
stopper = EarlyStopper(patience=15, min_delta=0.01)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
numEpochs = 200
batch_size = 32
data_division = [0.7, 0.2, 0.1] #70% train, 20% validation, 10% test

print(device)

dataset = TetrisDataset("boards", binaryBoard=True)
trainer = Trainer(model, criterion, optimizer, dataset, early_stopper=stopper)
trainer.to(device)
trainer.load_dataset(dataset, data_division)
trainer.train(numEpochs, batch_size)
trainer.save_model("model.pth")
trainer.plot()
trainer.evaluate()


