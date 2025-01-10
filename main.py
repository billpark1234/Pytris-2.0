import pygame
from gameComponents.render import Renderer
from gameComponents.field import Field
from gameComponents.controller import Controller
from extractor import Extractor
from gameEngine import GameEngine


class Main:
    def __init__(self, mode='play', displayPath=None):
        self.mode = mode
        self.displayPath = displayPath
    
    
    def run(self):
        if self.mode == 'play':
            self.play()
        elif self.mode == 'display':
            if self.displayPath is None:
                self.displayPath = input("Path to predefined boards (json):  ")
            self.display(self.displayPath)
    
    
    """
    Display predefined sequences of Tetris boards.
    """
    def display(self, path):
        from gameComponents.piece import SHAPE_ID
        renderer = Renderer()
        extractor = Extractor()
        field = Field()
        extractor.load(path)
        instances, labels = extractor.extractExamples()
        renderer.reset()
        field.reset()
        
        
        i = 0
        running = True
        while running:
            field.board = extractor.extendedRaw[i]
            renderer.render_frame(field, None, None, None, None, None, None)
            
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_LEFT:
                        #get prev board
                        i = max(0,i-1)
                        print("board index = " + str(i))
                        print("piece: " + SHAPE_ID(instances[i][1]).name)
                        print("x, rotation: " + str(labels[i][0]) + ", " + str(labels[i][1]))
                        print("\n")

                        
                    if event.key == pygame.K_RIGHT:
                        #get next board
                        i = min(len(extractor.getAll())-1,i+1)
                        print("board index = " + str(i))
                        print("piece: " + SHAPE_ID(instances[i][1]).name)
                        print("x: " + str(labels[i][0]) + ", rotation: " + str(labels[i][1]))
                        print("\n")
            
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
    main = Main(mode='display', displayPath="boards/3.json")
    main.run()