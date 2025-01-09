import json
import numpy as np
from window_settings import *

class Extractor:
    def __init__(self):
        self.raw = None
        self.extended = []
    
    def load(self, path):
        with open(path, 'r') as file:
            java_list = json.load(file)        
            self.raw = [np.array(arr) for arr in java_list]
            
            for board in self.raw:
                self.extended.append(np.vstack((np.zeros(shape=(NHIDDEN_ROWS, NCOLS), dtype=int), board)))
            
            
    def getBoard(self, i):
        assert 0 <= i < len(self.raw)
        return self.extended[i]
    
    
    def getAll(self):
        return self.extended
    
    
    """
    Extracts instance-label from raw dataset.
    An instance is a board and a piece. Label is the position and orientation of the piece.
    Position of a piece is the global coordinate of the center of the bounding box.
    The coordinate starts as (0,0) for top-left.
    """
    def extractExamples(self):
        pass