
import torch
from copy import deepcopy
import warnings
from numpy import floor
import torch.nn.functional as F
from torch.distributions import MultivariateNormal
import math

class CONTROL_2_VALUE(torch.nn.Module):
    """
    POLICY CLASS: CONTAINS ALL USABLE FUNCTIONAL FORMS FOR THE AGENTS POLICY, ALONG WITH
    THE CORROSPONDING HELPER FUNCTIONS (CONTINUOUS ACTIONS)
    """
    def __init__(self, state_size, action_size, hidden_layer_size, action_box = None, diag = True):
        super(CONTROL_2_VALUE, self).__init__()
        # dimentionality initializations
        self.state_size = state_size
        self.action_size = action_size
        # set the types
        self.state_type = None
        self.action_type = None
        # something to ensure that we dont take the log of zero
        self.epsilon = torch.tensor(0.1)
        # layer info
        self.linear1 = torch.nn.Linear(state_size, hidden_layer_size)
        self.linear2 = torch.nn.Linear(hidden_layer_size, 1)
        # last hidden layer
        self.tanh_1 = torch.nn.Tanh()
        self.tanh_2 = torch.nn.Tanh()

    def forward(self, state):
        # input
        input = torch.FloatTensor(state)
        # first layer
        baseline = self.linear1(input)
        baseline = self.tanh_1(baseline)
        baseline = self.linear2(baseline)
        baseline = self.tanh_2(baseline)
        # return it
        return baseline
