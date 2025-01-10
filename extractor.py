import json
import numpy as np
from window_settings import *
from gameComponents.piece import TILE_COORDINATES_X, TILE_COORDINATES_Y, SHAPE_ID

class InferenceFailedException(Exception):
    def __init__(self, board):
        self.board = board
        
    def getBoard(self):
        return self.board

"""
This array stores the translation offsets to map a tight bounding box to an actual bounding box, assuming that the tight bounding box
sits at the bottom-left of actual bounding box.

Example 1: Translating T-mino using Down translation
[0,6,0]       [0,0,0]
[6,6,6]  - >  [0,6,0]
.             [6,6,6]

Example 2: Translating standing up S-mino using right translation
[5,0]      [0,5,0]
[5,5]  - > [0,5,5]
[0,5]      [0,0,5]
"""
JLSZT_translationDefsAssumeBottomLeft = np.array([
    [0, 1], 
    [1, 0],
    [0,0]
])

I_translationDefsAssumeBottomLeft = np.array([
    [1, 2], 
    [2, 0],
])

class Extractor:
    def __init__(self):
        self.raw = None # raw boards are 2D array representation of pictures taken.
        self.extendedRaw = [] # extended boards are 2D array representation of raw boards with additional rows at the top to fit the specification for field.
        
    def load(self, path):
        with open(path, 'r') as file:
            java_list = json.load(file)        
            self.raw = [np.array(arr) for arr in java_list]
            
            for board in self.raw:
                self.extendedRaw.append(self.extend(board))
            

    
    """
    Extracts instance-label from *raw* dataset.
    An instance is an extended board and a piece. Label is the position and orientation of the piece.
    Position of a piece is the global coordinate of the center of the bounding box.
    The coordinate starts as (0,0) for bottom-left
    """
    def extractExamples(self):
        numBoards = len(self.raw)
        instances = []
        labels = []
        exampleIndex = 0
        
        for i in range(numBoards - 1):
            currBoardExtended = self.extend(self.truncateTop(self.raw[i]))
            nextBoardExtended = self.extend(self.truncateTop(self.raw[i+1]))
            pieceType = self.getPieceTypeValue(self.raw[i])
            
            if self.tileCount(currBoardExtended) < self.tileCount(nextBoardExtended):
                p, x, y, rt = self.inferPieceAttributes(currBoardExtended, nextBoardExtended, False)
                instances.append((self.removeGhost(currBoardExtended), p))
                labels.append((x, rt))
            else: #line clear occurred
                p, x, y, rt = self.inferPieceAttributes(currBoardExtended, nextBoardExtended, True)
                instances.append((self.removeGhost(currBoardExtended), p))
                labels.append((x, rt))

                
                
        return instances, labels
    
    
    """
    Infers the piece's position and its orientation at the time of placement on the current board.
    The input boards must be extended boards.
    
    Infers rotation in these steps: 1. get global tile coordinates. 2. Convert them into local coordinates. 3. Compare against predefined coordinates
    If line clear occurred, it's very difficult to infer just by comparing differences. In this case,
    I choose to infer using the ghost piece. It still has a very very low chance to be misleading.
    """
    def inferPieceAttributes(self, curr, next, inferByGhostPiece):
        assert curr.shape == next.shape
        
        delta = None
        ys = None
        xs = None
        pieceType = None
        if not inferByGhostPiece:
            delta = self.removeGhost(next) - self.removeGhost(curr)

            # get the global coordinates of tiles
            ys, xs = np.nonzero(delta)
            pieceType = delta[ys[0]][xs[0]]
            assert len(ys) == 4 and len(xs) == 4
        else:
            ys, xs = np.where(curr > 10)
            pieceType = curr[ys[0]][xs[0]] - 10 # ghost piece is represented by a value of piece type + 10
            assert len(ys) == 4 and len(xs) == 4
            
        assert ys is not None
        assert xs is not None
        assert pieceType is not None
        
        
        ys = NTOTAL_ROWS - ys - 1 ## convert index into coordinate
        
        # get the global coordinate of the bottom-left of the tight bounding box
        bottom = min(ys) 
        top = max(ys)
        left = min(xs)
        right = max(xs)
        
        # subtract bottom-left from all tile coords to get the local coords under tight bounding box.
        xs = xs - left
        ys = ys - bottom
        
        
        #Translate up, down, left, right
        inferenceSuccess = False
        translationDefs = I_translationDefsAssumeBottomLeft if pieceType == SHAPE_ID.I.value else JLSZT_translationDefsAssumeBottomLeft
        for translation in translationDefs:
            localXs = xs + translation[0]
            localYs = ys + translation[1]
            
            for rt in range(4):
                xdef = TILE_COORDINATES_X[pieceType - 1][rt]
                ydef = TILE_COORDINATES_Y[pieceType - 1][rt]
                if np.array_equiv(localXs, xdef) and np.array_equiv(localYs, ydef):
                    #offsetting tight global coordinate by bottom left of tile coords to get the proper x,y coordinate compatible with rotation
                    return pieceType, left - min(xdef), bottom - min(ydef), rt
        

    """
    Returns a copy of board whose top third rows are replaced with zero-rows
    """
    def truncateTop(self, board):
        copy = board.copy()
        for i in range(3):
            copy[i] = np.zeros(shape=(NCOLS), dtype=int)
            
        return copy
    
    
    """
    Returns a copy of board whose ghost piece is removed.
    """
    def removeGhost(self, board):
        copy = board.copy()
        for row in copy:
            row[np.where(row > 10)] = 0
        return copy
    
    
    """
    Counts the number of tiles in board
    """
    def tileCount(self, board):
        return np.count_nonzero(board)
    
    
    """
    Stack two empty rows at the top of a given raw board.
    Returns a new board
    """
    def extend(self, board):
        return np.vstack((np.zeros(shape=(NHIDDEN_ROWS, NCOLS), dtype=int), board))
            
    
    """
    Returns the height of a board. Height is the tallest column
    """
    def getHeight(self, board):
        n = len(board)
        for i in range(n):
            if np.count_nonzero(board[i]) > 0: #finds the index of the first row that has a nonzero element
                return n - i
    
    
    """
    Returns a board with cleared lines restored.
    Requires:
        The boards are extended boards not the raw ones
        
        
    TODO: Fix this. It does not work when the fell piece is not fully consumed for lineclear.
    """
    def restoreClearedLines(self, curr, next, pieceType):
        newBoard = np.zeros(shape=(NTOTAL_ROWS, NCOLS), dtype=int)
        curr = curr.copy()
        next = next.copy()
        i = NTOTAL_ROWS - 1
        j = NTOTAL_ROWS - 1
        k = NTOTAL_ROWS - 1
        
        while i >= 0 and j >= 0:
            currRow = curr[i]
            nextRow = next[j]
            
            if np.count_nonzero(currRow) == 0 or np.count_nonzero(nextRow) == 0:
                break
            elif np.array_equiv(currRow, nextRow):
                newBoard[k] = nextRow
                k -= 1
                i -= 1
                j -= 1
            else: #currRow must be a cleared row OR nextRow minus residue
                indices = np.nonzero(currRow == 0)
                currRow[indices] = pieceType
                newBoard[k] = currRow
                k -= 1
                i -= 1
 
        return newBoard
    
    
    """
    Returns the value of piece shape id given a *raw* board
    """
    def getPieceTypeValue(self, board):
        return board[0][np.nonzero(board[0])][0] ## first nonzero element of the top row
    
    
    """
    Returns a copy of list of boards on which two empty rows have been stacked.
    """
    def getAll(self):
        return self.extendedRaw.copy()
    
    
    """
    Get ith example extracted.
    An example is instance-label tuple where
    an instance is a numpy array with shape=(NTOTAL_ROWS, NCOLS) and the falling tetromino
    a corresponding label is the x coordinate and rotation of the tetromino at the placement
    """
    def getExample(self, i):
        return self.instances[i], self.labels[i]
            