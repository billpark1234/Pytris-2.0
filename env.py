import gymnasium as gym
import numpy as np
from gameEngine import GameEngine
from gameComponents.render import Renderer
from window_settings import *

class TetrisEnvironment(gym.Env):
    metadata = {"render_mode": ['human', 'field_array']}
    
    def __init__(self, render_mode=None):
        assert render_mode in self.metadata["render_mode"]
        
        if render_mode == "human":
            self.observation_space = gym.spaces.Dict({
                
                "piece": gym.spaces.Discrete(7, start=1),
                "coordinate": gym.spaces.Box(low=np.array([-2,-2]), high=np.array([NCOLS-1, NTOTAL_ROWS-1]), dtype=int), #(x,y)
                "orientation": gym.spaces.Discrete(4, start=0),
                "field": gym.spaces.Box(low=0, high=8, shape=(NTOTAL_ROWS, NCOLS), dtype=int),
                
            })
        elif render_mode == "field_array":
            self.observation_space = gym.spaces.Box(low=0, high=8, shape=(NTOTAL_ROWS, NCOLS), dtype=int)
        
        self.action_space = gym.spaces.MultiBinary(7)
        
        self.game = GameEngine(agent=None, render_mode=render_mode)
    
    
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.game.reset()
        obs, reward, terminated, truncated, info = self.game.get_obs()
        
        return obs, info
    
    
    def step(self, action):
        self.game.ctrl.copyPressedFrom(action)
                
        self.game.update()
        
        obs, reward, terminated, truncated, info = self.game.get_obs()
        
        return obs, reward, terminated, truncated, info
        
    
        
    def close(self):
        self.game.close()