
""" IMPORT PACKAGES """
import torch
import torch.multiprocessing as mp
import os
from copy import deepcopy
import warnings
from numpy import floor
import datetime
from torch.utils.data import DataLoader,TensorDataset

""" IMPORT CONFIGURATIONS """
from config import *

""" IMPORT MY OTHER PROGRAMS """
# import objective functions
from objective_functions.MEPG.objective import *
from objective_functions.RW.objective import *

# import grid world programs
from enviroments.grid_world.generate_reward_surface import *
from enviroments.grid_world.deterministic_grid_world import *

# load in agent model programs
from nn_architectures.agent_models.discrete_policies.grid_world_1_policy import *
from nn_architectures.agent_models.discrete_policies.grid_world_2_policy import *
from nn_architectures.agent_models.discrete_policies.grid_world_3_policy import *
from nn_architectures.agent_models.discrete_policies.grid_world_4_policy import *

# load in additional optimizers
from variance_reduction.trust_region_regularization.trust_region_methods import *
from optimizers.TRPO import *

class TRAIN_AGENT_GRID_WORLD(torch.nn.Module):

    """ GENERAL TRAIN AGENT CLASS: USED COELESCE INFORMATION IN ORDER TO TRAIN
        AGENT FOLLOWING KL(P||Q) SCHEME. INCLUDES ALL SPECS GIVEN IN THE
        SETTINGS FILE, WHERE EACH OF THESE PARAMETERS CAN BE ADJUSTED. """

    def __init__(self, config_path, il_agent_policy):
        super(TRAIN_AGENT_GRID_WORLD, self).__init__()

        """ LOAD CONFIGURATION FILE """
        config_dict = load_my_yaml(config_path)
        self.config = config_dict
        self.il_agent = il_agent_policy

        """ MODELS USED IN THE GAME """
        # set the type of network the agent will use
        self.agent_model = config_dict["AGENT_MODEL"]
        self.objective_type = config_dict["OBJECTIVE_TYPE"]

        """ GENERAL PARAMETERS """
        # load state and action dims
        self.state_size = config_dict["STATE_SIZE"]
        self.action_size = config_dict["ACTION_SIZE"]
        self.actions = config_dict["ACTIONS"]
        # set the number of iterations to train the agent
        self.iterations = config_dict["ITERATIONS"]
        # set the number of samples used per epo
        self.sample_size = config_dict["SAMPLE_SIZE"]
        # number of samples used per update
        self.batch_size = config_dict["BATCH_SIZE"]
        # set the length of the time horaizon considered
        self.trajectory_length = config_dict["TRAJECTORY_LENGTH"]
        # should we track mean gradient variance
        self.include_variance_calc = config_dict["INCLUDE_VARIANCE_CALC"]
        # set the reward shaping stuff
        self.inv_reward_shaping = eval(config_dict["INV_REWARD_SHAPING"])
        self.reward_shaping = eval(config_dict["REWARD_SHAPING"])

        """ OPTIMIZER HYPER-PARAMETERS """
        # set the type of optimizer we want to use
        self.optimize = config_dict["OPTIMIZE"]
        # set the type of optimizer we want to use
        self.hidden_layer_size = config_dict["HIDDEN_LAYER_SIZE"]
        # set hyper parameters
        self.lr = config_dict["LEARNING_RATE"]
        self.beta_1 = config_dict["BETA_1"]
        self.beta_2 = config_dict["BETA_2"]
        self.alpha = config_dict["ALPHA"]
        self.weight_decay = config_dict["WEIGHT_DECAY"]
        self.lambd = config_dict["LAMBDA"]
        self.gamma = config_dict["GAMMA"]
        # set step size decay intervals
        self.step_interval = config_dict["STEP_INTERVAL"]
        self.line_search = False
        # set mini-batch training info
        self.workers = config_dict["WORKERS"]

        """ ENVIROMENT PARAMETERS """
        # set grid info if we are in this game type
        self.grid_type = config_dict["GRID_TYPE"]
        self.grid_size = config_dict["GRID_SIZE"]
        self.stochasticity = config_dict["STOCHASTICITY"]

        """ CHECK IF THERE WE USE A PRE_CONDITIONER """
        if config_dict["PRECONDITION"]:
            self.rescale_point = True
            self.precondition = config_dict["PRECONDITION"]
            self.starting_step_size = config_dict["STARTING_STEP_SIZE"]
            self.divergence_limit = config_dict["DIVERGENCE_LIMIT"]
        else:
            self.rescale_point = False

    def set_optimizer(self, policy):

        """ INITIALIZE CHOSEN OPTIMIZER """
        if self.optimize == "Adam":
            optimizer = torch.optim.Adam(policy.parameters(), lr = self.lr, betas = (self.beta_1, self.beta_2),
                    weight_decay = self.weight_decay)
            return optimizer
        elif self.optimize == "ASGD":
            optimizer = torch.optim.ASGD(policy.parameters(), lr = self.lr, lambd=self.lambd, alpha = self.alpha,
                    weight_decay = self.weight_decay)
            return optimizer
        elif self.optimize == "SGD":
            optimizer = torch.optim.SGD(policy.parameters(), lr = self.lr, weight_decay = self.weight_decay)
            return optimizer
        elif self.optimize == "NSGD":
            self.rescale_point = True
            self.line_search = False
            self.trust_region = TRUST_REGION_PROGRAMS(self.config)
            optimizer = TRPO(policy.parameters(), lr = self.lr, weight_decay = self.weight_decay)
            return optimizer
        elif self.optimize == "TRPO":
            self.rescale_point = True
            self.line_search = True
            self.trust_region = TRUST_REGION_PROGRAMS(self.config)
            optimizer = TRPO(policy.parameters(), lr = self.lr, weight_decay = self.weight_decay)
            return optimizer
        else:
            raise Exception('optimization procedure not supported.')

    def set_policy(self):

        """ SET THE NUERAL NETWORK MODEL USED FOR THE CURRENT PROBLEM """
        if self.agent_model == 'GRID_WORLD_1_POLICY':
            return GRID_WORLD_1_POLICY(self.state_size, self.action_size, self.actions, self.hidden_layer_size)
        elif self.agent_model == 'GRID_WORLD_2_POLICY':
            return  GRID_WORLD_2_POLICY(self.state_size, self.action_size, self.actions, self.hidden_layer_size)
        elif self.agent_model == 'GRID_WORLD_3_POLICY':
            return  GRID_WORLD_3_POLICY(self.state_size, self.action_size, self.actions, self.hidden_layer_size)
        elif self.agent_model == 'GRID_WORLD_4_POLICY':
            return  GRID_WORLD_4_POLICY(self.state_size, self.action_size, self.actions, self.hidden_layer_size)
        else:
            raise Exception('policy not supported.')

    def set_loss(self):

        """ SET THE NUERAL NETWORK MODEL USED FOR THE CURRENT PROBLEM """
        if self.objective_type == 'MEPG':
            return MEPG_OBJECTIVE(self.trajectory_length, self.batch_size, self.config)
        elif self.objective_type == 'RW':
            return RW_OBJECTIVE(self.trajectory_length, self.batch_size, self.config)
        else:
            raise Exception('objective not supported.')

    def get_gradients(self,model):

        """ SOMETHING TO GRAB GRADIENTS TO TRACK VARIANCE THROUGH ITERATIONS """
        grads = []
        for param in model.parameters():
            grads.append(param.grad.reshape(-1).reshape(-1))
        return torch.cat(grads)

    def init_enviroment(self):
        # pick the reward
        reward_surface = REWARD_SURFACE()
        # initialize the grid
        grid = reward_surface.generate_reward_surface('RL_1')
        # set start state
        enviroment = DETERMINISTIC_GRID_WORLD(grid, [0,4])
        # return the enviroment
        return enviroment

    def train_agent(self):

        """ INITIALIZE PRIMARY INTERACTION ENVIROMENT """
        enviroment = self.init_enviroment()

        """ INFO TRACKING """
        # loss tracking
        loss_per_iteration = torch.zeros((self.iterations))
        # sample variance
        var_per_iteration = torch.zeros((self.iterations))

        """ INITIALIZE OBJECTIVE, POLICY, AND OPTIMIZER """
        # add model
        policy = self.set_policy()
        # add loss module
        loss = self.set_loss()
        # optimization method
        optimizer = self.set_optimizer(policy)
        # set learning rate scheduler
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=self.step_interval, gamma=self.gamma)

        """ TRAIN AGENT """
        for iter in range(self.iterations):

            """ SAMPLE FROM SIMULATOR """
            # get new batch of trajectories from simulator (sample with trained policy)
            state_total, action_total, reward_total = enviroment.simulate_trajectory(self.trajectory_length, self.sample_size, policy)
            # get new batch of trajectories from simulator (sample with il policy)
            il_state, il_action, _ = enviroment.simulate_trajectory(self.trajectory_length, self.sample_size, self.il_agent)

            """ COMPUTE LOSS AND BACK PROPOGATE """
            # compute loss
            loss_eval = loss(policy, state_batch, action_batch, reward_batch)
            # zero the parameter gradients
            optimizer.zero_grad()

            """ IF THE GRADIENT WILL BE SCALED """
            if self.rescale_point:
                # call .backwords to generate gradient update
                loss_eval.backward()
                # compute precondition matrix
                H = self.trust_region.compute_preconditioner(policy, state_batch, action_batch)
                # check if we are doing trust region
                if self.line_search:
                    # compute step before step
                    step_size = self.trust_region.back_tracking_linesearch(H, policy, (state_batch, action_batch, reward_batch), loss)
                    # step optimizer
                    optimizer.step(H, step_size)
                else:
                    # step optimizer
                    optimizer.step(H)
            else:
                # backprop through computation graph
                loss_eval.backward()
                # step optimizer
                optimizer.step()

            # set scheduler
            lr_scheduler.step()

            """ UPDATE DATA STORAGE """
            # update loss
            loss_per_iteration[iter] = torch.sum(torch.sum(self.inv_reward_shaping(reward_total),1)/self.sample_size)

            """ PRINT STATEMENTS """
            if iter % floor((self.iterations+1)/100) == 0:
                print("=========================================================")
                print("Current Time: " + str(datetime.datetime.now()))
                print(self.grid_size + " " + self.grid_type + " grid with " + str(100*self.stochasticity) + " percent stochasticity.")
                print("Expected Sum of rewards: " + str(loss_per_iteration[iter]))
                print("Loss: " + str(loss_eval))
                print("Percent complete: " + str(floor(100*iter/self.iterations)))

        """ RETURN """
        print("complete!")
        print("=========================================================")
        # return the trained policy and the loss per iteration
        return policy, loss_per_iteration, var_per_iteration
