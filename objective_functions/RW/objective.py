
# imports
import torch
import time
import torch.nn.functional as F


# import replay buffer stuff
from memory.replay_buffer import *

class RW_OBJECTIVE(torch.nn.Module):

    """
        REWEIGHTED WAKE SLEEP ALGORITHM: INFERENCE NETWORK WAKE PHASE.
        IN THIS STEP, EXACTLY THE SAME AS THE MODEL FREE CASE, WE TAKE
        THE TRAJECTORIES GENERATED UNDER THE POLICY AND SIMULATOR AND THEN
        TRAIN THE AGENT FOLLOWING THE IMPORTANCE WEIGHTING SCHEME DESCRIBED
        THEREIN.
    """

    """ INITIALIZATIONS """
    def __init__(self, trajectory_length, sample_size, config):
        super(RW_OBJECTIVE, self).__init__()
        self.sample_size = sample_size
        self.trajectory_length = trajectory_length
        # set the values we need from the config
        self.include_buffer = config['INCLUDE_BUFFER']
        self.buffer_size = config['BUFFER_SIZE']
        # initialize the actual buffer object
        self.replay_buffer = REPLAY_BUFFER(trajectory_length, sample_size, config)
        # set buffer init status
        self.buffer_init = False
        # set running average of marginal log-liklihood
        self.marginal_log_liklihood = None
        self.log_liklihood_shift = None
        # set averaging iterations
        self.avg_iters = torch.tensor(0.)

    """ LOSS FUNCTION FOR REINFORCEMENT LEARNING AS IMPORTANCE WEIGHTED APPROXIMATE INFERENCE FOLLOWING KL(P||Q) """
    def forward(self, current_policy, new_states, new_actions, new_rewards):

        # check if we will include a replay buffer + SIS
        if self.include_buffer and self.buffer_init:

            """ GET THE CURRENT SAMPLES FOR THE UPDATE SET UP """
            # now set the policy as a mixture between the two quanities
            policy = self.replay_buffer.generate_mixture_distribution(current_policy)
            # set the states and actions used for update
            state_tensor = torch.cat([new_states.float(), self.replay_buffer.buffer_states])
            action_tensor = torch.cat([new_actions.float(), self.replay_buffer.buffer_actions])
            reward_tensor = torch.cat([new_rewards.float(), self.replay_buffer.buffer_rewards])

            """ NEED A BLOCK WISE POLICY EVALUATOR """
            iw, _ = self.replay_buffer.compute_buffer_iw(policy, state_tensor, action_tensor, reward_tensor)
            _, score = self.replay_buffer.batch_iw(current_policy, state_tensor, action_tensor, reward_tensor)

            """ RETURN IT ALL """
            return -1*torch.dot(iw, score.reshape(-1))

        else:
            """ IF NOT APPLY EVERYTHING AS PER USUAL """
            policy = current_policy
            state_tensor = new_states.float()
            action_tensor = new_actions.float()
            reward_tensor = new_rewards.float()

            """ NEED A BLOCK WISE POLICY EVALUATOR """
            iw, score = self.replay_buffer.batch_iw(policy, state_tensor, action_tensor, reward_tensor)

        """ UPDATE THE REPLAY BUFFER (IF USED) """
        if self.include_buffer:
            # update buffer with the new samples
            self.replay_buffer.update_buffer(new_states, new_actions, new_rewards)
            # now set the buffer as initialized
            self.buffer_init = True

        """ RETURN IT ALL """
        return -1*torch.dot(iw, score.reshape(-1))
