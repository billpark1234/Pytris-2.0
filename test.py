class Test:
    def __init__(self):
        pass
    
    
    def extractorTest(self):
        from extractor import Extractor
        extractor = Extractor()
        extractor.load('boards/4.json')
        print(extractor.getBoard(5))
        

    def clearLineTest(self):
        from gameComponents.render import Renderer
        from gameComponents.field import Field
        import pygame
        import numpy as np
        
        f = Field()
        f.board[10] = np.array([2,3,3,3,2,0,3,0,2,2])
        f.board[18] = np.array([2,2,2,2,2,0,0,0,2,2])
        f.board[19] = np.array([1,1,1,1,1,1,1,1,1,1])
        f.board[20] = np.array([0,2,0,2,0,2,0,2,0,2])
        f.board[21] = np.array([1,1,1,1,1,1,1,1,1,1])
    
        
        r = Renderer()
        r.reset()
        
        running = True
        while running:
            
            r.render_frame(f.board, None, None, None)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break;
                
                if event.type == pygame.KEYDOWN:
                    f.clearLine()

        
        r.close()
        
        
# t = Test()
# t.clearLineTest()

import numpy as np
from gameComponents.piece import *

p = Piece(SHAPE_ID.I)
a = TILE_COORDINATES_X[0].copy()
print(a[0])