
import torch
import copy
from torch.optim.optimizer import Optimizer, required
from torch.nn.utils.convert_parameters import parameters_to_vector, vector_to_parameters

# import objective functions
from objective_functions.MEPG.objective import *
from objective_functions.RW.objective import *
# optimizer
from optimizers.TRPO import *


class TRUST_REGION_PROGRAMS(object):
    """docstring for TRUST_REGION_PROGRAMS."""
    def __init__(self, config_dict):
        super(TRUST_REGION_PROGRAMS, self).__init__()
        self.config = config_dict
        self.starting_step_size = config_dict["STARTING_STEP_SIZE"]
        self.init_step_size = config_dict["STARTING_STEP_SIZE"]
        self.divergence_limit = config_dict["DIVERGENCE_LIMIT"]
        self.optimize = config_dict["OPTIMIZE"]
        self.lr = config_dict["LEARNING_RATE"]
        self.weight_decay = config_dict["WEIGHT_DECAY"]
        self.objective_type = config_dict["OBJECTIVE_TYPE"]
        self.trajectory_length = config_dict["TRAJECTORY_LENGTH"]
        self.batch_size = config_dict["BATCH_SIZE"]

    def set_optimizer(self, policy):
        """ INITIALIZE CHOSEN OPTIMIZER """
        optimizer = TRPO(policy.parameters(), lr = self.lr, weight_decay = self.weight_decay)
        return optimizer

    def back_tracking_linesearch(self, H, policy, loss_info, loss):

        """ INIT EVERYTHING """
        # initial step size
        step_size = self.starting_step_size
        # duplicate the policy
        new_policy = copy.deepcopy(policy)
        # set the temp optim method
        optimizer = self.set_optimizer(new_policy)
        # get info to compute loss
        state_batch, action_batch, reward_batch = loss_info
        # flag to expand stepsize
        expand = False
        # """ CHECK IF THE CURRENT LOSS USES A REPLAY BUFFER """
        # if loss.include_buffer and loss.buffer_init:
        #     # set the states and actions used for update
        #     state_tensor = torch.cat([state_batch.float(), loss.replay_buffer.buffer_states])
        #     action_tensor = torch.cat([action_batch.float(), loss.replay_buffer.buffer_actions])

        # compute step size that satifies the KL divergence constraint
        while True:
            # get loss
            loss_eval = loss(new_policy, state_batch, action_batch, reward_batch)
            # take a step
            optimizer.zero_grad()
            loss_eval.backward()
            optimizer.step(H, step_size)
            # compute KL
            KL = self.compute_KL(new_policy, policy, state_batch, action_batch)
            # check if we have satisfied constraint
            if KL > self.divergence_limit:
                # duplicate the policy
                new_policy = copy.deepcopy(policy)
                # set the temp optim method
                optimizer = self.set_optimizer(new_policy)
                # half step size
                step_size = step_size / 2
            elif step_size < 1e-8:
                # update step size
                self.starting_step_size = step_size * 2
                break
            else:
                # set expansion flag
                if (not expand) and (self.init_step_size == step_size):
                    expand = True
                    break
                # if we hit the same starting step size twice, expand.
                elif expand and (self.init_step_size == step_size):
                    expand = False
                    self.starting_step_size = step_size * 2
                    break
                else:
                    expand = False
                    self.starting_step_size = step_size / 2
                    break
        # check that we are not exploding
        if 1e5*self.init_step_size < step_size:
            self.starting_step_size = self.init_step_size

        # return the trust region step size
        return step_size

    def compute_KL(self, new_policy, old_policy, states, actions):

        """ CONVERT FORMAT """
        flat_states = torch.flatten(states, start_dim=0,end_dim=1)
        flat_actions = torch.flatten(actions, start_dim=0,end_dim=1)

        """ COMPUTE COMPUTE MONTE-CARLO APPRIMXATION TO KL UNDER OLD POLICY"""
        divergence = -new_policy(flat_states,flat_actions)+old_policy(flat_states,flat_actions)
        divergence = torch.sum(divergence) / flat_states.size()[0]

        return divergence

    def compute_preconditioner(self, policy, state_tensor, action_tensor, current_loss):

        # """ CHECK IF THE CURRENT LOSS USES A REPLAY BUFFER """
        # if current_loss.include_buffer and current_loss.buffer_init:
        #     # if so, add in the buffer states to the precompute stuff
        #     state_tensor = torch.cat([state_tensor.float(), current_loss.replay_buffer.buffer_states])
        #     action_tensor = torch.cat([action_tensor.float(), current_loss.replay_buffer.buffer_actions])

        """ CONVERT FORMAT """
        flat_states = torch.flatten(state_tensor, start_dim=0,end_dim=1)
        flat_actions = torch.flatten(action_tensor, start_dim=0,end_dim=1)

        """ COMPUTE FIRST STEP """
        # create copy
        policy_copy = copy.deepcopy(policy)
        # evaluate loss
        score = policy_copy(flat_states[0,:],flat_actions[0,:])
        # step
        score.backward()
        # get gradients
        grad_i = parameters_to_vector([p.grad for p in policy_copy.parameters()])
        # take outer product
        H = torch.ger(grad_i, grad_i)
        # delete copy
        del policy_copy

        """ STEP THROUGH DATA AND BUILD FISHER INFO """
        for i in range(1, action_tensor.size()[0]):
            # create copy
            policy_copy = copy.deepcopy(policy)
            # zero the parameter gradients
            policy_copy.zero_grad()
            # evaluate loss
            score = policy_copy(flat_states[i,:],flat_actions[i,:])
            # step
            score.backward()
            # get gradients
            grad_i = parameters_to_vector([p.grad for p in policy_copy.parameters()])
            # take outer product
            H = torch.ger(grad_i, grad_i)
            # delete copy
            del policy_copy

        """ STABALIZE FOR USE LATER """
        preconditioner = H / action_tensor.size()[0]
        preconditioner += torch.tensor(1e-4)*torch.eye(preconditioner.size()[0])

        """ RETURN THE PRECONDITIONED MATRIX """
        return preconditioner
