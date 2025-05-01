import pygame
from gameComponents.render import Renderer
from gameComponents.field import Field
from gameComponents.controller import Controller
from extractor import Extractor
from gameEngine import GameEngine


class Main:
    def __init__(self, mode='play', displayPath=None, agent=None):
        self.mode = mode
        self.displayPath = displayPath
        self.agent = agent
    
    
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
        renderer = Renderer(render_mode="human")
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
            renderer.render_frame(field, None, None, None, None, None, None, None)
            
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
                        print("x: " + str(labels[i][0]) + ", rotation: " + str(labels[i][1]))
                        print("\n")

                        
                    if event.key == pygame.K_RIGHT:
                        #get next board
                        i = min(len(instances)-1,i+1)
                        print("board index = " + str(i))
                        print("piece: " + SHAPE_ID(instances[i][1]).name)
                        print("x: " + str(labels[i][0]) + ", rotation: " + str(labels[i][1]))
                        print("\n")
            
        renderer.close()
    
    
    """
    Initialize Tetris environment and play by yourself.
    """
    # def play(self):        
    #     game = GameEngine(self.agent)
    #     game.reset()
        
    #     running = True
    #     while running:
    #         for event in pygame.event.get():
    #             if event.type == pygame.QUIT:
    #                 running = False
    #                 break;
                
    #             if event.type == pygame.KEYDOWN or event.type == pygame.KEYUP:
    #                 if event.key == pygame.K_r:
    #                     game.close()
    #                     game.reset()
    #                 else:
    #                     game.ctrl.handleEvent(event)
                
    #         game.update()
                
                
    #     game.close()
        
        
    def play(self):       
        game = GameEngine(render_mode="human", agent=self.agent)
        game.reset()
        
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    break;
                
                if event.type == pygame.KEYDOWN or event.type == pygame.KEYUP:
                    if event.key == pygame.K_r:
                        game.reset()
                    else:
                        game.ctrl.handleEvent(event)
                
            game.update()
                
                
        game.close()

        
        

if __name__ == "__main__":
    from agents import CNN, CNN2, VisionTransformer
    import torch
    
    PATH = 'model.pth'
    num_layers = 8
    emb_size = 32
    num_head = 4
    num_class= 48
    patch_size=2
    net = VisionTransformer(num_layers=num_layers, img_size=20, emb_size=emb_size, patch_size=patch_size, num_head=num_head, num_class=num_class)
    net.load_state_dict(torch.load(PATH, weights_only=True))
    
    main = Main(mode='play', displayPath="boards/19.json", agent=net)
    main.run()
