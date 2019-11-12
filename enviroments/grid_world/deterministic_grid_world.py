
import torch
from copy import deepcopy
import warnings
from numpy import floor

class DETERMINISTIC_GRID_WORLD():
    """
    GRID WORLD CLASS: HELPER CLASS TO INSTANTIATE A SMALL MDP PROBLEM TO TEST POLICY GRADIENT
    METHODS ON TO ENSURE CORRECTNESS, ALSO ALOWS FOR FOR TESTING OF NEW TYPES OF ENVIROMENTS
    WITH DIFFERENT STATESPACE DISTRIBUTIONS ECT.
    """

    def __init__(self, pre_determined_grid, start_state):
        super(DETERMINISTIC_GRID_WORLD, self).__init__()
        self.grid = pre_determined_grid
        self.start_state = start_state
        self.grid_dim = [len(pre_determined_grid),len(pre_determined_grid[0])]

    def step_agent(self, state, action):
        #[row, column]
        if action == 1: # down
            # if we are in top row no change
            if state[0] == 0:
                None
            # otherwise increase row position
            else:
                state[0] -= 1
        elif action == 2: # up
            if state[0] == self.grid_dim[1]-1:
                None
            # otherwise increase row position
            else:
                state[0] += 1
        elif action == 3: # right
            if state[1] == self.grid_dim[0]-1:
                None
            # otherwise increase row position
            else:
                state[1] += 1
        elif action == 4: # down
            if state[1] == 0:
                None
            # otherwise increase row position
            else:
                state[1] -= 1
        elif action == 0: # dont_move
            None
        else:
            print("key error for movement in step_agent")
        self.current_state = state
        return state

    def simulate_trajectory(self, T, samples, policy):
        # initialize tensor storage
        state_tensor = torch.zeros((samples, T, len(self.start_state)), requires_grad=False)
        reward_tensor = torch.zeros((samples, T), requires_grad=False)
        action_tensor = torch.zeros((samples, T), requires_grad=False)
        # iterate through all states and actions
        for sample in range(samples):
            # initialize variables to iterate
            current_state =  deepcopy(self.start_state)
            current_action = policy.sample_action(current_state)
            # update state action
            action_tensor[sample,0] = current_action
            state_tensor[sample,0,:] = torch.FloatTensor(current_state)
            # simulate new values
            current_state = self.step_agent(current_state, current_action)
            current_action = policy.sample_action(current_state)
            # update reward
            reward_tensor[sample,0] = deepcopy(self.grid[current_state[0]][current_state[1]])
            # iterate over remaining time steps
            for t in range(1,T):
                # update state action
                action_tensor[sample,t] = current_action
                state_tensor[sample,t,:] = torch.FloatTensor(current_state)
                # simulate new values
                current_state = self.step_agent(current_state, current_action)
                current_action = policy.sample_action(current_state)
                # update reward
                reward_tensor[sample,t] = deepcopy(self.grid[current_state[0]][current_state[1]])
        # return it all
        return state_tensor, action_tensor.unsqueeze(2), reward_tensor.unsqueeze(2)
