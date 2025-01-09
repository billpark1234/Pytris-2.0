import pygame
from gameComponents.render import Renderer
from gameComponents.field import Field
from gameComponents.controller import Controller
from extractor import Extractor
from gameEngine import GameEngine


class Main:
    def __init__(self, mode='play'):
        self.mode = mode
    
    
    def run(self):
        if self.mode == 'play':
            self.play()
        elif self.mode == 'display':
            path = input("Path to predefined boards (json):  ")
            self.display(path)
    
    
    """
    Display predefined sequences of Tetris boards.
    """
    def display(self, path):
        renderer = Renderer()
        extractor = Extractor()
        field = Field()
        extractor.load(path)
        renderer.reset()
        field.reset()
        
        i = 0
        running = True
        while running:
            field.board = extractor.getBoard(i)
            renderer.render_frame(field, None, None, None)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        #get prev board
                        i = max(0,i-1)

                        
                    if event.key == pygame.K_RIGHT:
                        #get next board
                        i = min(len(extractor.getAll())-1,i+1)
            
        renderer.close()
    
    
    """
    Initialize Tetris environment and play by yourself.
    """
    def play(self):
        game = GameEngine()
        game.reset()
        
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break;
                
                if event.type == pygame.KEYDOWN or event.type == pygame.KEYUP:
                    if event.key == pygame.K_r:
                        game.close()
                        game.reset()
                    else:
                        game.ctrl.handleEvent(event)
                
            game.update()
                
                
        game.close()

        
        

if __name__ == "__main__":
    main = Main(mode='play')
    main.run()