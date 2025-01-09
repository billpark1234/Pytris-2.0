"""
Renderer is responsible for rendering the game.
Rendering structure follows this: window <- canvas <- features.

"""
import pygame
import numpy as np
from window_settings import *
from tetromino_settings import *
from gameComponents.piece import Piece
from gameComponents.field import Field
from gameComponents.piece import SHAPES_DEFAULT


class Renderer:
    def __init__(self, render_mode="human", fps=60):
        self.render_mode = render_mode
        self.fps = 60
        
        
    def reset(self):
        pygame.init()
        pygame.display.init()
        
        self.window = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.clock = pygame.time.Clock()
        self.canvas = pygame.Surface((WINDOW_WIDTH, WINDOW_HEIGHT))
        self.board_surface = pygame.Surface((VISIBLE_BOARD_WIDTH, VISIBLE_BOARD_HEIGHT))
        self.preview_surface = pygame.Surface((PREVIEW_WIDTH, PREVIEW_HEIGHT))
        self.hold_surface = pygame.Surface((HOLD_WIDTH, HOLD_WIDTH))
        self.stat_surface = None


    """
    For each frame, the renderer does the following:
    Flush, Draw, Blit, Display.
    
    Requires that board is a numpy array of shape=(NHIDDEN_ROWS + NROWS, NCOLS)
    """
    def render_frame(self, field, pieceQueue, holdPiece, piece, x, y, ghostCoordinates):            
        assert self.window is not None 
        
        #Render the components on canvas
        self.flushAll()
        self.drawBoard()
        self.drawPieces(field)
        self.drawFallingPiece(x, y, piece, field)
        self.drawPreview(pieceQueue)
        self.drawHold(holdPiece)
        self.drawGhost(piece, ghostCoordinates, field)
        self.blitAll()
        
        
        #Render canvas on window
        if self.render_mode == 'human':
            pygame.event.pump()
            self.clock.tick(self.fps)
            self.window.fill(0)
            self.window.blit(self.canvas, (0,0))
            pygame.display.flip()
            
        elif self.render_mode == "rgb_array":
            return np.transpose(
                np.array(pygame.surfarray.pixels3d(self.window)), axes=(1,0,2)
            )
        
        
    def drawBoard(self):
        points = [(0,0),(VISIBLE_BOARD_WIDTH-2,0),(VISIBLE_BOARD_WIDTH-2,VISIBLE_BOARD_HEIGHT-2),(0,VISIBLE_BOARD_HEIGHT-2)]
        pygame.draw.lines(self.board_surface, BORDER_GRAY, True, points, 2)
        
        
    def drawPieces(self, field : Field):
        for x in range(0, field.getWidth()):
            for y in range(0, field.getVisibleHeight()): #iterate thru visible rows
                if(field.getValue(x,y) != 0):
                    tile = pygame.Surface((CELL_SIZE, CELL_SIZE))
                    tile.fill(field.getColor(x,y))
                    self.board_surface.blit(tile, (x*CELL_SIZE, (y * -1 + field.getVisibleHeight() - 1)*CELL_SIZE))
        
    #x,y is global coordinate
    def drawFallingPiece(self, x, y, piece : Piece, field : Field):
        if piece is not None:
            for i in range(4):
                local_x = piece.tileCoordX[piece.orientation][i]
                local_y = piece.tileCoordY[piece.orientation][i]
                
                tile = pygame.Surface((CELL_SIZE, CELL_SIZE))
                tile.fill(piece.getColor())
                self.board_surface.blit(tile, ((x + local_x) * CELL_SIZE, ((y + local_y) * -1 + field.getVisibleHeight() - 1)*CELL_SIZE))
    
    
    def drawHold(self, holdPiece):
        if holdPiece is not None:
            pieceArray = SHAPES_DEFAULT[holdPiece]
            for x in range(pieceArray.shape[1]):
                for y in range(pieceArray.shape[0]):
                    if pieceArray[y,x] != 0:
                        tile = pygame.Surface((CELL_SIZE, CELL_SIZE))
                        tile.fill(TETROMINO_COLORS[holdPiece.value])
                        self.hold_surface.blit(tile, ((x+1)*CELL_SIZE, (y+1)*CELL_SIZE))
                        
    
    def drawGhost(self, piece: Piece, ghostCoordinates, field: Field):
        if ghostCoordinates is not None:
            x, y = ghostCoordinates
            for i in range(4):
                local_x = piece.tileCoordX[piece.orientation][i]
                local_y = piece.tileCoordY[piece.orientation][i]
                
                tile = pygame.Surface((CELL_SIZE, CELL_SIZE))
                tile.fill(piece.getColor())
                tile.set_alpha(128)
                self.board_surface.blit(tile, ((x + local_x) * CELL_SIZE, ((y + local_y) * -1 + field.getVisibleHeight() - 1)*CELL_SIZE))
                
    
    def drawPreview(self, pieceQueue):
        for i in range(len(pieceQueue)):
            next_surf = self._render_next(pieceQueue[i])
            self.preview_surface.blit(next_surf, (CELL_SIZE, (3*i+1)*CELL_SIZE))
    
    
    def _render_next(self, type):
        piece_array = SHAPES_DEFAULT[type]
        width = piece_array.shape[1]
        height = piece_array.shape[0]
        next_surf = pygame.Surface((width*CELL_SIZE, height*CELL_SIZE))
        for px in range(width):
            for py in range(height):
                if(piece_array[py,px] != 0):
                    tile = pygame.Surface((CELL_SIZE, CELL_SIZE))
                    tile.fill(TETROMINO_COLORS[type.value])
                    next_surf.blit(tile, (px * CELL_SIZE, py * CELL_SIZE))
        return next_surf
    
    
    def blitAll(self):
        self.canvas.blit(self.board_surface, (MARGIN + HOLD_WIDTH, MARGIN))
        self.canvas.blit(self.preview_surface, (VISIBLE_BOARD_WIDTH + MARGIN + HOLD_WIDTH, MARGIN))
        self.canvas.blit(self.hold_surface, (0, MARGIN))
        
        
    def flushAll(self):
        self.preview_surface.fill(0)
        self.board_surface.fill(0)
        self.hold_surface.fill(0)
        
        
    ## needs to be placed in env.py
    def close(self):
        if self.window is not None:
            pygame.display.quit()
            pygame.quit()