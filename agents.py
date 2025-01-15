
import torch.nn as nn
import torch.nn.functional as F
import torch
import gymnasium as gym

"""
Input : board of shape (batch size, in channel, height, width) = (1, 1, 22, 10) and an integer tensor (batch, 1) = (1, 1)
Output : tensor of (batch size, 48) = (1, 48)
"""
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 4, 4) #[1,22, 10] -> [4, 19, 7]
        self.pool1 = nn.MaxPool2d(2,2) #[4, 19, 7] -> [4, 9, 3]
        self.conv2 = nn.Conv2d(4, 16, 2) #[4 9 3] -> [16 8 2]
        self.pool2 = nn.MaxPool2d(2, 2) # [16 8 2] -> [16 4 1]
        self.embedding = nn.Embedding(7, 5) #7 tetrominoes. embedding dim is chosen arbitrarily
        
        self.fc1 = nn.Linear(16*8*2 + 5, 130)
        self.fc2 = nn.Linear(130, 65)
        self.fc3 = nn.Linear(65, 48)
        
        
    def forward(self, board, piece):
        board = self.pool1(F.relu(self.conv1(board)))
        board = F.relu(self.conv2(board))
        board = torch.flatten(board, 1)
        
        mino = piece - 1 # piece takes in range (1..7) inclusive, but embedding index is from 0 to 6
        mino = self.embedding(mino)
        mino = torch.flatten(mino, 1)
        
        combined = torch.cat((board, mino), dim=1)

        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x



class CNN2(nn.Module):
    def __init__(self):
        super(CNN2, self).__init__()
        
        # self.conv1 = nn.Conv2d(1, 2, 2) #[1,22, 10] -> [2, 21, 9]
        self.embedding = nn.Embedding(7, 5) #7 tetrominoes. embedding dim is chosen arbitrarily
        
        self.fc1 = nn.Linear(1*22*10 + 5, 130)
        self.fc2 = nn.Linear(130, 65)
        self.fc3 = nn.Linear(65, 48)
        
        
    def forward(self, board, piece):
        board = torch.flatten(board, 1)
        
        mino = piece - 1 # piece takes in range (1..7) inclusive, but embedding index is from 0 to 6
        mino = self.embedding(mino)
        mino = torch.flatten(mino, 1)
        
        combined = torch.cat((board, mino), dim=1)

        x = F.relu(self.fc1(combined))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        
        return x



class RLAgent:
    def __init__(
        self,
        env: gym.Env,
        learning_rate: float,
        initial_epsilon: float,
        epsilon_decay: float,
        final_epsilon: float,
        discount_factor: float = 0.95,
    ):
        self.env = env
        
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.epsilon_decay = epsilon_decay
        self.final_epsilon = final_epsilon
        self.training_error = []
        
    
