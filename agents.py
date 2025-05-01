
import torch.nn as nn
import torch.nn.functional as F
import torch
import gymnasium as gym
from einops.layers.torch import Rearrange
from einops import repeat, rearrange
import numpy as np


def PositionEmbedding(seq_len, emb_size):
    embeddings = torch.ones(seq_len, emb_size)
    for i in range(seq_len):
        for j in range(emb_size):
            embeddings[i][j] = np.sin(i / (pow(10000, j / emb_size))) if j % 2 == 0 else np.cos(i / (pow(10000, (j - 1) / emb_size)))
    return torch.tensor(embeddings)


class PatchEmbedding(nn.Module):
    """
    Input: board of shape (batch size, H, W)
    Output: tensor of (batch size, num_patches, emb_size)
    """
    def __init__(self, in_channels: int = 1, patch_size: int = 2, emb_size: int = 32, img_size=20):
        self.patch_size = patch_size
        super().__init__()
        # (b, cin, H, W) -> (b, emb_size, H/patch_size, W/patch_size)
        # (b, emb_size, H/patch_size, W/patch_size) -> (b, num_patches, emb_size)
        self.embed = nn.Sequential(
            nn.Conv2d(in_channels, emb_size, kernel_size=patch_size, stride=patch_size),
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

        self.cls_token = nn.Parameter(torch.rand(1, 1, emb_size))
        self.pos_embed = nn.Parameter(PositionEmbedding((img_size // patch_size)**2 + 1, emb_size))

    def forward(self, x):
        b, _, _, _ = x.shape
        x = self.embed(x)    

        cls_token = repeat(self.cls_token, ' () s e -> b s e', b=b)

        x = torch.cat([cls_token, x], dim=1)

        x = x + self.pos_embed 
        return x


class MultiHead(nn.Module):
    def __init__(self, emb_size, num_head):
        super().__init__()
        self.emb_size = emb_size
        self.num_head = num_head
        self.key = nn.Linear(emb_size, emb_size)
        self.value = nn.Linear(emb_size, emb_size)
        self.query = nn.Linear(emb_size, emb_size) 
        self.att_dr = nn.Dropout(0.1)
    def forward(self, x):
        k = rearrange(self.key(x), 'b n (h e) -> b h n e', h=self.num_head)
        q = rearrange(self.key(x), 'b n (h e) -> b h n e', h=self.num_head)
        v = rearrange(self.key(x), 'b n (h e) -> b h n e', h=self.num_head)

        wei = q@k.transpose(3,2)/self.num_head ** 0.5    
        wei = F.softmax(wei, dim=2)
        wei = self.att_dr(wei)

        out = wei@v

        out = rearrange(out, 'b h n e -> b n (h e)')
        return out
    

class FeedForward(nn.Module):
    def __init__(self, emb_size):
        super().__init__()
        self.ff = nn.Sequential(
            nn.Linear(emb_size, 4*emb_size),
            nn.Linear(4*emb_size, emb_size)
        )
    def forward(self, x):
        return self.ff(x)
  

class Block(nn.Module):
    def __init__(self,emb_size, num_head):
        super().__init__()
        self.att = MultiHead(emb_size, num_head)
        self.ll =   nn.LayerNorm(emb_size)
        self.dropout = nn.Dropout(0.1)
        self.ff = FeedForward(emb_size)
    def forward(self, x):
        x = x + self.dropout(self.att(self.ll(x)))  # self.att(x): x -> (b , n, emb_size) 
        x = x + self.dropout(self.ff(self.ll(x)))
        return x
    

class VisionTransformer(nn.Module):
    def __init__(self, num_layers, img_size, emb_size, patch_size, num_head, num_class):
        super().__init__()
        self.attention = nn.Sequential(*[Block(emb_size, num_head) for _ in range(num_layers)])
        self.patchemb = PatchEmbedding(patch_size=patch_size, img_size=img_size)
        self.piece_embed = nn.Embedding(8, emb_size)
        self.ff = nn.Linear(emb_size, num_class)

    def forward(self, board, piece):     # x -> (b, c, h, w)
        embeddings = self.patchemb(board)     # (b, num_patches + 1, emb_size)

        # Add piece embedding to cls token
        piece_emb = self.piece_embed(piece)   # (b, 1, emb_size)
        # make piece_emb (b,1,emb_size) -> (b, emb_size)
        piece_emb = piece_emb.squeeze(1) 
        embeddings[:, 0, :] += piece_emb

        x = self.attention(embeddings)        # (b, num_patches+1, emb_size)
        x = self.ff(x[:, 0, :])                # Take the cls token only
        return x

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

