import numpy as np
from tetromino_settings import *
from window_settings import *


predefinedBoards = {
    "TSPIN_TRIPLE" : np.array([
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [1,1,0,0,0,0,0,0,0,0],
        [1,0,0,0,0,0,0,0,0,0],
        [1,0,1,1,1,1,1,1,1,1],
        [1,0,0,1,1,1,1,1,1,1],
        [1,0,1,1,1,1,1,1,1,1],
    ]),
    
    "TEST1" : np.array([
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0,0,0,0],
        [1,1,1,0,0,0,0,0,0,0],
        [1,1,0,0,0,0,0,0,0,0],
        [1,1,0,1,1,1,1,1,1,1],
        [1,1,0,0,1,1,1,1,1,1],
        [1,1,0,0,0,0,0,0,0,1],
        [1,1,0,0,0,0,0,0,1,1],
        [1,1,1,1,0,1,0,1,1,1],
        [1,1,1,0,0,1,0,0,1,1],
        [1,1,1,1,0,1,1,0,1,1],
        [1,1,0,1,1,1,1,1,1,1],
        [1,1,0,0,0,0,1,1,1,1],
        [1,1,0,0,0,0,1,1,1,1],
        [1,1,1,1,0,0,1,1,1,1],
        [1,1,1,1,1,0,1,1,1,1],
    ])
    
}

"""
Field is a wrapper for tetris board.
While a 2D array's coordinate starts with 0,0 at top-left, 
field's coordinate starts with 0,0 at bottom right, like cartesian.
"""
class Field:
    
    WALL_VALUE = 8
    
    def __init__(self):
        self.board = None
        
        #Flags indicate whether a line is full
        self.flags = None
        
    
    def reset(self):
        self.board = np.zeros(shape=(NHIDDEN_ROWS + NROWS, NCOLS), dtype=int)
        self.flags = np.full(len(self.board), False, dtype=bool)
        
    
    def clearLine(self):
        for i, row in enumerate(self.board):
            if np.all(row != 0):
                self.flags[i] = True
            
        remainder = np.zeros(shape=(len(self.board), len(self.board[0])), dtype=int)
        j = len(remainder) - 1
        
        for i in reversed(range(len(self.board))):
            if not self.flags[i]:
                remainder[j] = self.board[i].copy()
                j -= 1
                    
        self.board = remainder
        self.flags = np.full(len(self.board), False, dtype=bool)
        
    
    def greyfyBoard(self):
        for y in range(self.board.shape[0]):
            for x in range(self.board.shape[1]):
                if self.board[y][x] != 0:
                    self.board[y][x] = Field.WALL_VALUE
        
    
    def getValue(self, x, y):
        if y < 0 or y >= self.getHeight():
            return Field.WALL_VALUE
        if x < 0 or x >= self.getWidth():
            return Field.WALL_VALUE

        return self.board[self.getHeight() - y - 1][x]
    
    
    def setValue(self, x, y, val):
        self.board[self.getHeight() - y - 1][x] = val
    
    
    def getColor(self, x, y):
        return TETROMINO_COLORS[self.getValue(x,y)]
        
    
    def getBoard(self):
        return self.board.copy()
    
    
    def getWidth(self):
        return self.board.shape[1]
    
    
    def getHeight(self):
        return self.board.shape[0]
    
    
    def getVisibleHeight(self):
        return self.board.shape[0] - NHIDDEN_ROWS
    
    
    def setupPredefinedBoard(self):
        self.board = predefinedBoards["TEST1"]