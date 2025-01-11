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
            
            r.render_frame(f, None, None, None)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break;
                
                if event.type == pygame.KEYDOWN:
                    f.clearLine()

        
        r.close()
        
    
    def extractor_remove_ghost_test(self):
        from extractor import Extractor
        e = Extractor()
        e.load("boards/3.json")
        b = e.raw[0]
        print(b)
        b = e.removeGhost(b)
        print(b)
        



        
# t = Test()
# t.extractor_remove_ghost_test()

from extractor import Extractor
import numpy as np
from agents import CNN
import torch

e = Extractor()
e.load("boards/2.json")

ins, lab = e.extractExamples()
board = ins[0][0]
print(board.shape)

boardTensor = torch.tensor(board, dtype=torch.float32).unsqueeze(0)
pieceTensor = torch.tensor(1, dtype=torch.long).unsqueeze(0).unsqueeze(0)
print(boardTensor.shape)
print(pieceTensor.shape)