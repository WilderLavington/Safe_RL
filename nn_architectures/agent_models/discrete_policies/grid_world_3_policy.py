
import torch
from copy import deepcopy
import warnings
from numpy import floor
import torch.nn.functional as F
from torch.distributions import Categorical

class GRID_WORLD_3_POLICY(torch.nn.Module):
    """
    POLICY CLASS: CONTAINS ALL USABLE FUNCTIONAL FORMS FOR THE AGENTS POLICY, ALONG WITH
    THE CORROSPONDING HELPER FUNCTIONS
    """
    def __init__(self, state_size, action_size, actions, hidden_layer_size):
        super(GRID_WORLD_3_POLICY, self).__init__()
        # dimentionality initializations
        self.state_size = state_size
        self.action_size = action_size
        self.actions = actions
        # something to ensure that we dont take the log of zero
        self.epsilon = torch.tensor(0.0001)
        # layer info
        self.linear1 = torch.nn.Linear(state_size, hidden_layer_size)
        self.linear2 = torch.nn.Linear(hidden_layer_size, hidden_layer_size)
        self.linear3 = torch.nn.Linear(hidden_layer_size, actions)
        # output
        self.softmax = torch.nn.Softmax(dim=0)
        # add a stacked output for vector calc
        self.outputstacked = torch.nn.Softmax(dim=1)
        # distribution variable for sampling
        self.dist = lambda prob: Categorical(prob)

    def sample_action(self, state):
        input = torch.FloatTensor(state)
        probabilities = self.linear1(input)
        probabilities = torch.tanh(probabilities)
        probabilities = self.linear2(probabilities)
        probabilities = torch.tanh(probabilities)
        probabilities = self.linear3(probabilities)
        probabilities = self.softmax(probabilities)
        action = self.dist(probabilities).sample()
        # return action sampled from data
        return action

    def forward(self, state, action):
        input = torch.FloatTensor(state)
        action = action.long()
        probabilities = self.linear1(input)
        probabilities = torch.tanh(probabilities)
        probabilities = self.linear2(probabilities)
        probabilities = torch.tanh(probabilities)
        probabilities = self.linear3(probabilities)
        if action.size()[0] == 1:
            output = self.softmax(probabilities)
            return torch.log(output[action] + self.epsilon)
        else:
            output = self.outputstacked(probabilities)
            # return log probability of an action
            return torch.log(torch.gather(output, 1, action) + self.epsilon)
