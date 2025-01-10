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


import numpy as np

arr = np.array([[1,2,11],
               [1,2,11]])

print(np.where(arr > 10))