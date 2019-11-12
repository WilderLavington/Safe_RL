
# imports
import torch
import time

# import replay buffer stuff
from memory.replay_buffer import *

# import control variate stuff
from variance_reduction.control_variates.value_function_handler import *
from variance_reduction.cost2go_approximation.q_function_handler import *

# PG objective function
class MEPG_OBJECTIVE(torch.nn.Module):

    """ POLICY GRADIENTS LOSS FUNCTION """
    def __init__(self, trajectory_length, sample_size, config):

        """ INITIALIZATIONS """
        super(MEPG_OBJECTIVE, self).__init__()
        # set configs
        self.config = config
        # initialize basic parameters
        self.sample_size = sample_size
        self.trajectory_length = trajectory_length
        # set the values we need from the config
        self.include_buffer = config['INCLUDE_BUFFER']
        self.buffer_size = config['BUFFER_SIZE']
        # initialize the actual buffer object
        self.replay_buffer = REPLAY_BUFFER(trajectory_length, sample_size, config)
        # a check that the buffer is initialized.
        self.buffer_init = False
        # decay term
        self.decay = 1.0
        # set the control variate used for the scheme
        self.vf_manager = None
        # if q function is not none
        self.qf_manager = None

    def set_value_function_manager(self):
        # init a value function manager
        vf_manager = VALUE_FUNCTION_MANAGER(self.config)
        # initialize all the info for it
        vf_manager.init_value_function()
        # initialize value function
        self.vf_manager = vf_manager
        # nothing to return
        return None

    def set_q_function_manager(self):
        # init a value function manager
        qf_manager = Q_FUNCTION_MANAGER(self.config)
        # initialize all the info for it
        qf_manager.init_q_function()
        # initialize value function
        self.qf_manager = qf_manager
        # nothing to return
        return None

    def set_decay(self, decay):
        # set the decay to be something if you want to
        self.decay = decay
        # nothing to return
        return None

    def compute_entropy(self, current_policy, state, action, roll_out):
        # check dimentions of rollout
        roll_out_dims = roll_out.size()
        entropy = -1*current_policy(state,action)
        entropy_dims = entropy.size()
        if not len(roll_out_dims) == len(entropy_dims):
            entropy = entropy.unsqueeze(1)
        return self.decay*entropy

    def forward(self, current_policy, new_states, new_actions, new_rewards):

        # if we want to include a buffer
        if self.include_buffer and self.buffer_init:
            # now set the policy as a mixture between the two quanities
            policy = self.replay_buffer.generate_local_mixure_dist(current_policy)
            # set the states and actions used for update
            state_tensor = torch.cat([new_states.float(), self.replay_buffer.buffer_states])
            action_tensor = torch.cat([new_actions.float(), self.replay_buffer.buffer_actions])
            reward_tensor = torch.cat([new_rewards.float(), self.replay_buffer.buffer_rewards])
        # otherwise buisness as usual
        else:
            policy = current_policy
            state_tensor = new_states.float()
            action_tensor = new_actions.float()
            reward_tensor = new_rewards.float()

        """ COMPUTE CUMULATIVE ROLL-OUT """
        cumulative_rollout = torch.zeros(reward_tensor.size())
        cumulative_rollout[:,self.trajectory_length-1] = reward_tensor[:,self.trajectory_length-1]
        for t in reversed(range(self.trajectory_length-1)):
            cumulative_rollout[:,t] = reward_tensor[:,t] + cumulative_rollout[:,t+1]
            cumulative_rollout[:,t] += self.compute_entropy(current_policy, state_tensor[:,t,:],action_tensor[:,t], cumulative_rollout[:,t])

        """ CONVERT FORMAT """
        flat_states = torch.flatten(state_tensor, start_dim=0,end_dim=1)
        flat_actions = torch.flatten(action_tensor, start_dim=0,end_dim=1)
        flat_cumsum = torch.flatten(cumulative_rollout, start_dim=0,end_dim=1)

        """ CALCULATE LIKLIHOOD """
        logliklihood_tensor = current_policy(flat_states,flat_actions)

        """ CALCULATE ADVANTAGE (MC) """
        A_hat = -flat_cumsum

        """ SUBTRACT BASELINE """
        if self.vf_manager is not None:
            A_hat -= self.vf_manager.value_function(flat_states)

        """ IF WE ARE USING A BUFFER APPLY IMPORTANCE WEIGHT SCALING """
        if self.include_buffer and self.buffer_init:
            # compute importance weights
            iw = logliklihood_tensor.reshape(-1) - policy(flat_states,flat_actions)
            iw = torch.exp(iw).detach()
            # now scale the advantage terms
            A_hat = A_hat.reshape(-1)*iw
            # now re-detach advantage
            A_hat = A_hat.detach()
        else:
            A_hat = A_hat.reshape(-1)
            # now detach
            A_hat = A_hat.detach()

        """ UPDATE THE REPLAY BUFFER (IF USED) """
        if self.include_buffer:
            # update buffer with the new samples
            self.replay_buffer.update_buffer(new_states, new_actions, new_rewards)
            # now set the buffer as initialized
            self.buffer_init = True

        """ UPDATE THE CONTROL VARIATE IF ITS NOT NULL """
        if self.vf_manager is not None:
            self.vf_manager.step_value_function(self.qf_manager.q_function, policy, new_states, new_actions, new_rewards)

        """ UPDATE THE COST TO GO IF ITS NOT NULL """
        if self.vf_manager is not None:
            self.qf_manager.step_q_function(self.vf_manager.value_function, policy, new_states, new_actions, new_rewards)

        """ CALCULATE POLICY GRADIENT OBJECTIVE """
        expectation = torch.dot(A_hat,logliklihood_tensor.reshape(-1))/self.trajectory_length

        """ RETURN """
        return expectation/self.sample_size
