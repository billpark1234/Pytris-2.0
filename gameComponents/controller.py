"""
Controller receives inputs
"""
import numpy as np
from enum import Enum
from pygame import KEYDOWN, KEYUP, K_LEFT, K_RIGHT, K_z, K_x, K_c, K_SPACE, K_UP, K_DOWN

class Controller:
    def __init__(self):
        self.key_mapping = {
            K_LEFT: 0,
            K_RIGHT: 1,
            K_z: 2,
            K_x: 3,
            K_DOWN: 4,
            K_SPACE: 5,
            K_c: 6,
        }
        self.reset()
        
    
    def reset(self):
        #[False, False, ... , False] representing pressed keys
        self.pressed = np.full(7, False, dtype=bool)
        
        #[0,0, ... ,0] number of frames each button is pressed for.
        self.buttonTime = np.zeros(7, dtype=int)
        
    
    """
    Updates pressed states upon receiving pygame's KEYDOWN or KEYUP event.
    """
    def handleEvent(self, event):
        if event.key in self.key_mapping:
            if event.type == KEYDOWN:
                self.pressed[self.key_mapping[event.key]] = True
            
            if event.type == KEYUP:
                self.pressed[self.key_mapping[event.key]] = False
       
        

    """
    Returns True if an initial button push is detected.
    """
    def isPush(self, btn):
        return self.buttonTime[self.key_mapping[btn]] == 1
    
    
    def isPress(self, btn):
        return self.buttonTime[self.key_mapping[btn]] >= 1
    
    
    def setButtonPressed(self, btn):
        self.pressed[self.key_mapping[btn]] = True
        
        
    def setButtonUnpressed(self, btn):
        self.pressed[self.key_mapping[btn]] = False 
        
    
    def getButtonTime(self, btn):
        return self.buttonTime[self.key_mapping[btn]]
    
    
    def copyPressedFrom(self, actions):
        assert isinstance(actions, np.ndarray)
        self.pressed = actions
        
        
    def updateTime(self):
        for i in range(len(self.buttonTime)):
            self.buttonTime[i] = self.buttonTime[i] + 1 if self.pressed[i] else 0
            

            
            
    