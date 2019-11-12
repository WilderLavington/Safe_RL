

import torch
from copy import deepcopy
import warnings
from numpy import floor

class REWARD_SURFACE():
    """
    GENERATES A REWARD SURFACE FOR EITHER THE IMITATION LEARNING AGENT,
    OR THE RL AGENT.
    """
    def __init__(self, size):
        super(REWARD_SURFACE, self).__init__()
        self.init = True

    def generate_rl_grid_1(self):
        # set the grid for true rl agent
        rl_grid = [[-1.0,  -1.0,  -1.0,   0.0,   1.0]
                   [-2.0,  -10.0, -2.0,  -1.0,  -1.0]
                   [-5.0,  -10.0, -4.0,  -10.0, -2.0]
                   [-8.0,  -10.0, -5.0,  -4.0,  -3.0],
                   [-10.0, -8.0,  -5.0,  -10.0, -10.0]]
        # convert to rl grid to tensor
        rl_grid = torch.tensor(rl_grid)
        # return em
        return rl_grid

    def generate_il_grid_1(self):
        # set the grid learned by imitation learning agent
        il_grid = [[-10.0, -10.0, -10.0, -10.0,  10.0]
                   [-10.0, -10.0, -10.0, -10.0,  1.0]
                   [-10.0, -10.0, -10.0, -10.0,  0.0]
                   [-10.0, -10.0, -3.0,  -2.0,  -1.0],
                   [-10.0, -5.0,  -4.0,  -10.0, -10.0]]
        # convert to rl grid to tensor
        il_grid = torch.tensor(il_grid)
        #return em
        return il_grid

    def generate_reward_surface(self, surface_type):
        # pick the type of reward surface
        if surface_type == 'RL_1':
            grid = self.generate_rl_grid_1()
        elif surface_type == 'IL_1':
            grid = self.generate_il_grid_1()
        # return the desired surface
        return grid
