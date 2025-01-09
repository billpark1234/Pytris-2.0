import numpy as np
from enum import Enum
from gameComponents.field import Field
from tetromino_settings import TETROMINO_COLORS
"""
Piece.py contains the following:

Enum class SHAPE_ID -- enums representing 7 tetrominos
SHAPES_DEFAULT -- a dict that maps shape id to its 2D array representation
ROTATED_SHAPES -- a dict that maps shape id and rotation to its 2D array representation
TILE_COORDINATES_X -- x coordinates of each of 4 tiles for each of 4 orientations for each of 7 tetrominos
TILE_COORDINATES_Y -- y coordinates of each of 4 tiles for each of 4 orientations for each of 7 tetrominos

class Piece -- a container class representing a tetromino piece. It manipulates the tile coordinates
                when a rotation is requested. Note that while orientation of the piece is stored in
                it self, the global (x,y) coordinate is stored in the game engine. This is because
                global coordinate does not change the relative tile coordinates
"""


class SHAPE_ID(Enum):
    I = 1
    J = 2
    L = 3
    O = 4
    S = 5
    T = 6
    Z = 7


SHAPES_DEFAULT = {
    SHAPE_ID.I : np.array(
        [[0,0,0,0,0],
        [0,0,0,0,0],
        [0,1,1,1,1],
        [0,0,0,0,0],
        [0,0,0,0,0]]),
    
    SHAPE_ID.J : np.array(
        [[2,0,0],
        [2,2,2],
        [0,0,0]]),
    
    SHAPE_ID.L : np.array(
        [[0,0,3],
        [3,3,3],
        [0,0,0]]),
    
    SHAPE_ID.O : np.array(
        [[0,4,4],
        [0,4,4],
        [0,0,0]]),
    
    SHAPE_ID.S : np.array(
        [[0,5,5],
        [5,5,0],
        [0,0,0]]),
    
    SHAPE_ID.T : np.array(
        [[0,6,0],
        [6,6,6],
        [0,0,0]]),
    
    SHAPE_ID.Z : np.array(
        [[7,7,0],
        [0,7,7],
        [0,0,0]]),
}

def generate_rotations(shape):
    return [np.rot90(shape, k=i, axes=(1,0)) for i in range(4)]


def getTileCoords(type):
    
    # 4 orientations, each has 4 tiles
    xcoords = np.zeros(shape=(4,4), dtype=int)
    ycoords = np.zeros(shape=(4,4), dtype=int)

    for i in range(4):
        xs, ys = getTileCoordsHelper(type, i)
        xcoords[i] = xs
        ycoords[i] = ys
        
    return xcoords, ycoords

"""
Returns a list of x coordinates and a list of y coordinates of tiles for rt orientation
"""
def getTileCoordsHelper(type, rt):
    shape = ROTATED_SHAPES[type][rt]
    # 4 tiles
    xs = np.zeros(shape=4, dtype=int)
    ys = np.zeros(shape=4, dtype=int)
    
    i = 0
    for y in range(len(shape)):
        for x in range(len(shape[0])):
            if shape[y][x] != 0:
                xs[i] = x
                ys[i] = len(shape) - y - 1
                i += 1
    
    return xs, ys



ROTATED_SHAPES = {shape_id: generate_rotations(shape) for shape_id, shape in SHAPES_DEFAULT.items()}
"""
ROTATED_SHAPES[type][orientation] is a 2D array representing a tetromino under the orientation index.
Orientation index increases clockwise. 0 is the default orientation.
"""



TILE_COORDINATES_X = np.zeros(shape=(7,4,4), dtype=int)
TILE_COORDINATES_Y = np.zeros(shape=(7,4,4), dtype=int)
"""
TILE_COORDINATES_X[type][rotation][i] is the x-coordinate of ith tile of a rotated tetromino.
TILE_COORDINATES_Y is for y-coordinates.
"""

for tetromino in SHAPE_ID:
    xcoords, ycoords = getTileCoords(tetromino)
    TILE_COORDINATES_X[tetromino.value - 1] = xcoords
    TILE_COORDINATES_Y[tetromino.value - 1] = ycoords



## 5 tests, 4 orientations, offset is a 2 dim vector.
JLSTZ_OFFSET_DATA = np.zeros(shape=(5,4,2), dtype=int)
JLSTZ_OFFSET_DATA[0,0] = np.array([0,0])
JLSTZ_OFFSET_DATA[0,1] = np.array([0,0])
JLSTZ_OFFSET_DATA[0,2] = np.array([0,0])
JLSTZ_OFFSET_DATA[0,3] = np.array([0,0])

JLSTZ_OFFSET_DATA[1,0] = np.array([0,0])
JLSTZ_OFFSET_DATA[1,1] = np.array([1,0])
JLSTZ_OFFSET_DATA[1,2] = np.array([0,0])
JLSTZ_OFFSET_DATA[1,3] = np.array([-1,0])

JLSTZ_OFFSET_DATA[2,0] = np.array([0,0])
JLSTZ_OFFSET_DATA[2,1] = np.array([1,-1])
JLSTZ_OFFSET_DATA[2,2] = np.array([0,0])
JLSTZ_OFFSET_DATA[2,3] = np.array([-1,-1])

JLSTZ_OFFSET_DATA[3,0] = np.array([0,0])
JLSTZ_OFFSET_DATA[3,1] = np.array([0,2])
JLSTZ_OFFSET_DATA[3,2] = np.array([0,0])
JLSTZ_OFFSET_DATA[3,3] = np.array([0,2])

JLSTZ_OFFSET_DATA[4,0] = np.array([0,0])
JLSTZ_OFFSET_DATA[4,1] = np.array([1,2])
JLSTZ_OFFSET_DATA[4,2] = np.array([0,0])
JLSTZ_OFFSET_DATA[4,3] = np.array([-1,2])


I_OFFSET_DATA = np.zeros(shape=(5,4,2), dtype=int)
I_OFFSET_DATA[0,0] = np.array([0,0])
I_OFFSET_DATA[0,1] = np.array([-1,0])
I_OFFSET_DATA[0,2] = np.array([-1,1])
I_OFFSET_DATA[0,3] = np.array([0,1])

I_OFFSET_DATA[1,0] = np.array([-1,0])
I_OFFSET_DATA[1,1] = np.array([0,0])
I_OFFSET_DATA[1,2] = np.array([1,1])
I_OFFSET_DATA[1,3] = np.array([0,1])

I_OFFSET_DATA[2,0] = np.array([2,0])
I_OFFSET_DATA[2,1] = np.array([0,0])
I_OFFSET_DATA[2,2] = np.array([-2,1])
I_OFFSET_DATA[2,3] = np.array([0,1])

I_OFFSET_DATA[3,0] = np.array([-1,0])
I_OFFSET_DATA[3,1] = np.array([0,1])
I_OFFSET_DATA[3,2] = np.array([1,0])
I_OFFSET_DATA[3,3] = np.array([0,-1])

I_OFFSET_DATA[4,0] = np.array([2,0])
I_OFFSET_DATA[4,1] = np.array([0,-2])
I_OFFSET_DATA[4,2] = np.array([-2,0])
I_OFFSET_DATA[4,3] = np.array([0,2])


O_OFFSET_DATA = np.zeros(shape=(1,4,2), dtype=int)
O_OFFSET_DATA[0,0] = np.array([0,0])
O_OFFSET_DATA[0,1] = np.array([0,-1])
O_OFFSET_DATA[0,2] = np.array([-1,-1])
O_OFFSET_DATA[0,3] = np.array([-1,0])



class Piece:

    # x,y is global coordinate of the center tile
    def __init__(self, type):
        self.type = type
        self.orientation = 0
        #X and Y coordinates for each tile of this piece. -1 is for translating ID value to index
        self.tileCoordX = TILE_COORDINATES_X[type.value - 1].copy()
        self.tileCoordY = TILE_COORDINATES_Y[type.value - 1].copy()
        self.offsets = None
        if type is SHAPE_ID.I:
            self.offsets = I_OFFSET_DATA
        elif type is SHAPE_ID.O:
            self.offsets = O_OFFSET_DATA
        else:
            self.offsets = JLSTZ_OFFSET_DATA
            
        self.offsetApplied = False
                        
    
    """
    Rotates the piece clockwise (1) / counter-clockwise (-1) / no rotation (0)
    """
    def rotate(self, dir, x, y, field):
        rotationSuccess = False
        currOrientation = self.orientation
        nextOrientation = self._mod(currOrientation + dir, 4)
        self.orientation = nextOrientation
        
        for i in range(5):
            currOffsets = self.offsets[i][currOrientation]
            nextOffsets = self.offsets[i][nextOrientation]
            offsetDelta = currOffsets - nextOffsets
            
            if not self.checkCollision(x + offsetDelta[0] , y + offsetDelta[1], field):
                rotationSuccess = True
                break;

        if not rotationSuccess:
            self.orientation = currOrientation
            
        return rotationSuccess, offsetDelta
        
    
    
    def bottomTileCoords(self):
        pass
    
    """
    Returns True if tetromino whose bottom-left corner's global coordinate is (x,y) collides with
    anything on the field.
    """
    
    def checkCollision(self, x, y, field: Field):
        for i in range(4):
            x2 = x + self.tileCoordX[self.orientation][i]
            y2 = y + self.tileCoordY[self.orientation][i]
            
            # print("evaluating " + str(i) + "th tile of " + str(self.type))
            # print("local tile coordinate (loc_x, loc_y) = " + str(self.tileCoordX[i]) + "," + str(self.tileCoordY[i]))
            # print("global piece coordinate = " + str(x) + "," + str(y))
            # print("global tile coordinate= " + str(x2) + "," + str(y2))

        
            if field.getValue(x2, y2) != 0:
                return True
            
        return False

        
    def applyOffsetX(self, offsetX):
        pass
    
    
    def getTileX(self):
        return self.tileCoordX[self.orientation].copy()
    
    
    def getTileY(self):
        return self.tileCoordY[self.orientation].copy()
    
    
    def getColor(self):
        return TETROMINO_COLORS[self.type.value]
    
    
    def getType(self):
        return self.type
    
    
    def ghostCoordinates(self, x, y, field: Field):
        while not self.checkCollision(x, y - 1, field):
            y -= 1
        
        return x, y
    """
    mod that works for positive and negative values
    x is the dividend, m is the divisor
    """
    def _mod(self, x, m):
        return (x % m + m) % m
    