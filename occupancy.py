

import torch
from torch.distributions.bernoulli import Bernoulli
from torch.distributions import Categorical
from copy import deepcopy
from numpy import floor
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import imageio

from torch.autograd import gradcheck

def empirical_dist_t(states, grid_dims):
    grid = torch.zeros(grid_dims)
    for i in range(len(states[0])):
        grid[states[0,i].long(), states[1,i].long()] += 1
    loc, mode = torch.mode(states)
    return grid #/ mode[0].float()

def generate_frames(state_samples, grid_dims, T):
    # iterate through all time steps and save png of emperical grid
    images = []
    for t in range(T):
        # generate grid
        grid_t = empirical_dist_t(state_samples[:,t,:], grid_dims)
        # convert to numpy for ease
        a = grid_t.numpy()
        # print(grid_t)
        # plot
        plt.imshow(a, cmap='hot', interpolation='nearest')
        plt.savefig('results/vizualizations/ila_occupancy_images/occ'+str(t)+'.png')
        images.append(imageio.imread('results/vizualizations/ila_occupancy_images/occ'+str(t)+'.png'))
        # plt.show()
        # save it to file
        # plt.savefig('occupancy_images/occ'+str(t)+'.png')
    imageio.mimsave('results/vizualizations/ila_occupancy_images/trajectory.gif', images)
