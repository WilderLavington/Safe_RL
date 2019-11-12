
# general imports
import torch
from copy import deepcopy
import warnings
from numpy import floor
import matplotlib.pyplot as plt
import argparse
import yaml
import os

# gym task imports
import gym

# import objective functions
from objective_functions.MEPG.objective import *
from objective_functions.RW.objective import *

# import grid world programs
from enviroments.grid_world.generate_reward_surface import *
from enviroments.grid_world.deterministic_grid_world import *
from enviroments.grid_world.stochastic_grid_world import *

# load in additional optimizers
from variance_reduction.trust_region_regularization.trust_region_methods import *
from optimizers.TRPO import *

# now import control
from model_training.classic_control.train_model import *

# load in model training programs
from model_training.grid_world.train_model import *

"""
THIS TEST WAS PRODUCED TO ANALYZE THE GRID WORLD ENVIROMENTS UNDER BOTH THE
RW AND MEPG OBJECTIVES FOR VARYING NETWORK SIZES, REPLAY BUFFERS, AND OPTIMIZERS
"""

def test():


    """ TRAIN IMITATION LEARNING AGENT: GRID WORLD SETUPS """
    # set the config path that has all the info you want the il agent to train
    config_path = 'config_files/grid_world/il_model_config.yaml'
    # set class
    training_algorithm = TRAIN_AGENT_GRID_WORLD(config_path)
    # get info
    il_policy, avg_loss_per_iteration = training_algorithm.train_agent()
    # generate a set of trajectories
    state_tensor, _, _  = training_algorithm.enviroment.simulate_trajectory(10, 1000, il_policy)
    # generate a example trajectory
    generate_frames(state_tensor, [4,0], 10)

    """ TRAIN RL LEARNING AGENT THAT IS CLOSE IN DIST TO IL AGENT """
    


    """ LOAD IN DATA STORAGE CONFIGURATIONS """
    config_dict = load_my_yaml(config_path)

    #save
    torch.save(avg_loss_per_iteration, config_dict["LOSS_DATA_STORAGE"]+"/loss_1.pt")
    torch.save(avg_var_per_iteration, config_dict["VARIANCE_DATA_STORAGE"]+"/variance_1.pt")

    # apply averaging
    for i in range(args.averaging_iterations-1):
        # get info
        policy, loss_per_iteration, var_per_iteration = training_algorithm.train_agent()
        #save
        torch.save(avg_loss_per_iteration, config_dict["LOSS_DATA_STORAGE"]+"/loss_"+str(i)+".pt")
        torch.save(avg_var_per_iteration, config_dict["VARIANCE_DATA_STORAGE"]+"/variance_"+str(i)+".pt")
        # add to averaging
        avg_loss_per_iteration += loss_per_iteration
        avg_var_per_iteration += var_per_iteration

    # now divide
    avg_loss_per_iteration /= args.averaging_iterations
    avg_var_per_iteration /= args.averaging_iterations

    # save the tesnors to resuls
    torch.save(avg_loss_per_iteration, config_dict["LOSS_DATA_STORAGE"]+"/loss.pt")
    torch.save(avg_var_per_iteration, config_dict["VARIANCE_DATA_STORAGE"]+"/variance.pt")

    # generate figure for loss
    fig1 = plt.figure()
    plt.plot(avg_loss_per_iteration.numpy())
    plt.title("Expected Return per epoch: " + config_dict["GRID_SIZE"] + ", " + config_dict["GRID_TYPE"] + ", " + config_dict["OBJECTIVE_TYPE"])
    plt.xlabel("Iteration")
    plt.ylabel("Expected Return")
    plt.savefig(config_dict["LOSS_PLOT_STORAGE"]+"/loss.png")

    # generate figure for variance
    fig2 = plt.figure()
    plt.plot(avg_var_per_iteration.numpy())
    plt.title("Variance per epoch: " + config_dict["GRID_SIZE"] + ", " + config_dict["GRID_TYPE"] + ", " + config_dict["OBJECTIVE_TYPE"])
    plt.xlabel("Iteration")
    plt.ylabel("Variance")
    plt.savefig(config_dict["VARIANCE_PLOT_STORAGE"]+"/variance.png")
