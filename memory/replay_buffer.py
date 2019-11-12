
# imports
import torch
import time
from torch.distributions.bernoulli import Bernoulli


# PG objective function
class REPLAY_BUFFER(torch.nn.Module):

    """ POLICY GRADIENTS LOSS FUNCTION """
    def __init__(self, trajectory_length, sample_size, config, chosen_buffer='probabalistic_greedy'):

        """ INITIALIZATIONS """
        super(REPLAY_BUFFER, self).__init__()
        # initialize basic parameters
        self.sample_size = sample_size
        self.trajectory_length = trajectory_length
        self.epsilon = torch.tensor(0.000001)
        # set the values we need from the config
        self.include_buffer = config['INCLUDE_BUFFER']
        self.buffer_size = config['BUFFER_SIZE']
        # flag to check if the buffer is empty
        self.buffer_initialized = False
        # set inits
        if self.include_buffer:
            # set of information required for re-sample and evaluation
            self.buffer_iw = None
            self.buffer_states = None
            self.buffer_actions = None
            self.buffer_rewards = None
            self.buffer_distribution = None
            # select the type of buffer
            self.chosen_buffer = self.init_buffer_type(chosen_buffer)
        # set current approx of marginal log likliood
        self.marginal_log_liklihood = None
        # something to scale how the buffer will regularized
        self.alpha = 0.01

    """ FUNCTION NEEDED FOR UNIFORMLY SAMPLING STATE, ACTION, REWARD FROM REPLAY BUFFER """
    def uniform_sample(self):
        return None

    """ HELPER TO UPDATE ALPHA DEPENDING ON WHAT ALGO WE USING """
    def set_alpha(self, alpha):
        self.alpha = alpha
        return None

    """ INITIALIZE BUFFER TYPE """
    def init_buffer_type(self, chosen_buffer):
        if chosen_buffer == 'greedy':
            return self.greedy_buffer
        elif chosen_buffer == 'unif':
            return self.uniform_buffer
        elif chosen_buffer == 'probabalistic_greedy':
            return self.probabalistic_greedy

    """ THIS GIVE IMPORTANCE WEIGHTS BETWEEN OPTIMAL TRAJECTORIES AND A POLICY (BATCH MODE)"""
    def batch_iw(self, policy, state_tensor, action_tensor, reward_tensor):

        """ POLICY SCORE FUNCTION LOG Q_OLD(TAU|O) """
        # convert format to something we can feed to model
        flat_states = torch.flatten(state_tensor, start_dim=0,end_dim=1)
        flat_actions = torch.flatten(action_tensor, start_dim=0,end_dim=1)
        # compute the models score function
        flat_opt_scoreFxn = policy(flat_states, flat_actions)
        # reshapte this tensor to be time by samples
        opt_scoreFxn = flat_opt_scoreFxn.reshape(reward_tensor.size())
        # sum accross time
        sum_opt_scoreFxn = torch.sum(opt_scoreFxn, dim=1)

        """ USE OPTIMALITY VARIABLES AS INDICATORS + COMPUTE LOG LOG P(TAU,O) """
        log_joint = torch.sum(reward_tensor, dim=1)

        """ COMPUTE BUFFER IMPORTANCE WEIGHTS """
        # sum through time to get the iw
        iw = log_joint.reshape(-1) - sum_opt_scoreFxn.reshape(-1)

        """ NORMALIZE IT """
        # stabalize numerically
        iw -= torch.max(iw)
        # compute exponential for the weights
        iw = torch.exp(iw) / torch.sum(torch.exp(iw))

        """ RETURN IT """
        return iw.detach(), sum_opt_scoreFxn

    """ THIS GIVE IMPORTANCE WEIGHTS BETWEEN OPTIMAL TRAJECTORIES AND A POLICY (NO BATCH MODE)"""
    def compute_buffer_iw(self, policy, state_tensor, action_tensor, reward_tensor):
        # set dimensions
        n,t,d = state_tensor.size()
        # compute joint
        log_joint = torch.sum(reward_tensor, dim=1)
        log_prop = torch.zeros(log_joint.size())
        # iterate through all samples
        for sample in range(n):
            log_prop[sample] = policy(state_tensor[sample,:,:], action_tensor[sample,:,:])
        # compute the log importance weights
        log_iw = log_joint.reshape(-1) - log_prop.reshape(-1)
        # normalize
        log_iw -= torch.max(log_iw)
        # normalize
        iw = torch.exp(log_iw) / torch.sum(torch.exp(log_iw))
        # detach from computation graph
        iw = iw.detach()
        # now return em
        return iw, log_prop

    """ RETURNS THE SCORE OF A TRAJECTORY UNDER THE CURRENT POLICY """
    def compute_score_function(self, policy, state_tensor, action_tensor):
        # compute the models score function
        opt_scoreFxn = policy(state_tensor, action_tensor).reshape(-1)
        # return summed through time
        return torch.sum(opt_scoreFxn,dim=0)

    """ CHECKS IF A TRAJECTORY IS IN REPLAY BUFFER """
    def replaybuffer_dist(self, states, actions):
        # get dimensions of buffer
        n,t,d = self.buffer_states.size()
        # iterate and check if any of the states match
        for i in range(n):
            condition_1 = torch.all(torch.eq(actions, self.buffer_actions[i,:]))
            condition_2 = torch.all(torch.eq(states, self.buffer_states[i,:,:]))
            if condition_1 and condition_2:
                # if it is in there return its importance weight
                return self.buffer_iw[i]
        # if it wasnt in there
        return torch.tensor(0.)

    """ COMPUTES THE MIXTURE PDF FROM CURRENT POLICY AND DISCRETE DISTRIBUTION """
    def generate_mixture_distribution(self, policy):
        # set mixture weights
        pi_0 = self.sample_size / (self.current_buffer_size + self.sample_size)
        pi_1 = 1-pi_0
        # set log probability under current policy
        proposal_prob = lambda states, actions: torch.exp(self.compute_score_function(policy, states, actions))
        # now create binary operator function to check if state / action is in buffer
        buffer_prob = lambda states, actions: self.replaybuffer_dist(states, actions)
        # now write out mixture
        mixture = lambda states, actions: pi_0*proposal_prob(states, actions) + pi_1*buffer_prob(states, actions)
        # now we really want the log mixture
        log_mixture = lambda states, actions: torch.log(mixture(states, actions) + self.epsilon)
        # return this function
        return log_mixture

    """ RETURNS THE PROBABILITY OF SAMPLING A STATE ACTION PAIR FROM THE BUFFER """
    def buffer_state_action_prob(self, hash_tensor, state_action):
        # check its not outside the dimentions of the tensor
        inside = tuple(state_action.tolist()) in hash_tensor.keys()
        # if they are, grab the element from the hash map
        if inside:
            return torch.tensor([hash_tensor[tuple(state_action.tolist())]])
        else:
            return torch.tensor([0.])

    """ GENERATE JOINT AND MARGINAL DISTRIBUTIONS USING FULL TRAJECTORIES """
    def create_trajectory_level_dist(self, states, actions, importance_weight):

        """ NOW CREATE LOOK UP """
        # get size of actions
        action_size = actions.size()[-1]
        # get unique stuff
        unique_state_action, state_action_map, state_action_counts = torch.unique(torch.cat([states.float(), actions.float()], dim=1).t(),\
                                    return_counts=True,\
                                    return_inverse=True,\
                                    dim=1, sorted=True)
        unique_state, state_map, state_counts = torch.unique(states.float().t(),\
                                    return_counts=True, \
                                    return_inverse=True, \
                                    dim=1, sorted=True)

        # re- transpose them to make them easier to work with
        unique_state = unique_state.t()
        unique_state_action = unique_state_action.t()

        """ HASH THE STATES AND ACTIONS INTO DICTIONARIES """
        joint_prob_hash = {}
        marginal_prob_hash = {}
        state_action_list = unique_state_action.tolist()
        state_list = unique_state.tolist()
        for indx, _ in enumerate(state_action_list):
            joint_prob_hash[tuple(state_action_list[indx])] = {}
            joint_prob_hash[tuple(state_action_list[indx])]['val'] = importance_weight*state_action_counts[indx].float()/self.trajectory_length
            joint_prob_hash[tuple(state_action_list[indx])]['state'] = tuple(state_action_list[indx][:-action_size])
        for indx, _ in enumerate(state_list):
            marginal_prob_hash[tuple(state_list[indx])] = {}
            marginal_prob_hash[tuple(state_list[indx])]['val'] = importance_weight*state_counts[indx].float()/self.trajectory_length

        """ RETURN THE DICTIONARIES """
        return joint_prob_hash, marginal_prob_hash

    """ ADDS VALUES IN DICTIONARIES """
    def combine_dictionaries(self, main_dict, new_dict):
        # add values from one dictionary to another
        for key in new_dict.keys():
            if key in main_dict.keys():
                main_dict[key]['val'] += new_dict[key]['val']
            else:
                main_dict[key] = {}
                main_dict[key]['val'] = new_dict[key]['val']
                if 'state' in new_dict[key].keys():
                    main_dict[key]['state'] = new_dict[key]['state']
        return main_dict

    """ CREATES A MIXTURE OVER STATE ACTION LEVEL PROBABILITIES"""
    def generate_local_mixure_dist(self, current_policy):

        """ CONVERT FORMAT """
        flat_states = torch.flatten(self.buffer_states, start_dim=0,end_dim=1)
        flat_actions = torch.flatten(self.buffer_actions, start_dim=0,end_dim=1).float()

        """ ITERATE THROUGH ALL TRAJECTORIES AND COMPUTE THEIR DISTRIBUTIONS AND ADD TO TOTAL """
        marginal_prob_dictionary = {}
        joint_prob_dictionary = {}
        for sample in range(self.current_buffer_size):
            # set the trajectory we are adding information over
            iw = self.buffer_iw[sample]
            states = self.buffer_states[sample,:,:]
            actions = self.buffer_actions[sample,:,:]
            # get the scaled prob of trajectories
            joint_prob_hash, marginal_prob_hash = self.create_trajectory_level_dist(states, actions, iw)
            # now add these to the main dictionaries
            joint_prob_dictionary = self.combine_dictionaries(joint_prob_dictionary, joint_prob_hash)
            marginal_prob_dictionary = self.combine_dictionaries(marginal_prob_dictionary, marginal_prob_hash)

        """ COMPUTE THE BUFFER PROBS FOR ALL STATE ACTIONS AS HASH MAP """
        # create dictionary
        buffer_prob_dict = {}
        for key in joint_prob_dictionary.keys():
            buffer_prob_dict[key] = joint_prob_dictionary[key]['val'] / marginal_prob_dictionary[joint_prob_dictionary[key]['state']]['val']

        """ USE THIS TENSOR TO CREATE A FUNCTION THAT EMITS PROB OF THAT TRANSITION IN BUFFER """
        buffer_prob = lambda flat_state_actions: torch.cat([self.buffer_state_action_prob(buffer_prob_dict, state_action.float()) for state_action in flat_state_actions])

        """ COMPUTE THE MIXTURE """
        # set mixture weights
        pi_0 = self.sample_size / (self.current_buffer_size + self.sample_size)
        pi_1 = 1-pi_0
        # define batch mixture function
        local_mixture_model = lambda flat_states_, flat_actions_: pi_0*torch.exp(current_policy(flat_states_, flat_actions_)).reshape(-1)\
                                                        +pi_1*buffer_prob(torch.cat([flat_states_.float(), flat_actions_.float()], dim=1))
        # really we want the log prob
        log_local_mixture_model = lambda flat_states_, flat_actions_: torch.log(local_mixture_model(flat_states_, flat_actions_))

        # return it
        return log_local_mixture_model

    """ COMPUTE GREEDY IMPORTANCE WEIGHTING SCHEME BASED UPON REWARDS PER TRAJECTORY """
    def greedy_buffer(self, rewards, resample_flag = False):
        iw = torch.sum(rewards, dim=1)
        iw = iw - torch.max(iw)
        # this is added to interpolate between uniform sampling and greedy
        iw = (1-self.alpha)*iw
        # buisness as usual
        iw = torch.exp(iw)
        iw = iw.reshape(-1)
        iw = iw / torch.sum(iw)
        return iw

    """ COMPUTE UNIFORM IMPORTANCE WEIGHTING SCHEME BASED UPON REWARDS PER TRAJECTORY """
    def uniform_buffer(self, rewards, resample_flag = False):
        s,d,r = rewards.size()
        iw = (1/s) * torch.ones(s)
        return iw

    """ PARTIALLY GREEDY APPROACH WHERE SAMPLES ARE DROPPED AT EACH UPDATE PROPTO THEIR RANK ^ ALPHA """
    def probabalistic_greedy(self, rewards, resample_flag = False):

        # if we are not re-sampling during this call
        if not resample_flag:
            iw = torch.sum(rewards, dim=1)
            iw = iw - torch.max(iw)
            iw = torch.exp(iw)
            iw = iw.reshape(-1)
            iw = iw / torch.sum(iw)
            return iw
        # otherwise
        else:
            # set the current iw
            iw = torch.sum(rewards, dim=1)
            iw = iw - torch.max(iw)
            iw = torch.exp(iw)
            iw = iw.reshape(-1)
            iw = iw / torch.sum(iw)
            # set up scaled bernoulli
            next_iw_dist = Bernoulli(iw**self.alpha)
            # set iw to zero with prob iw^2
            iw = next_iw_dist.sample()*iw
            # rescale everything
            iw = iw / torch.sum(iw)
            # return everything
            return iw

    """ REMOVE DUPLICATES AND ENSURE THAT WE HAVE ALL NON-ZERO IMPORTANCE """
    def clean_buffer(self):

        """ REMOVE DUPLICATE TRAJECTORIES """
        tau = torch.cat([self.buffer_states, self.buffer_actions, self.buffer_rewards], dim=2)
        unique_tau = torch.unique(tau, dim=0)
        self.buffer_states = unique_tau[:,:,:self.buffer_states.size()[2]]
        self.buffer_actions = unique_tau[:,:,self.buffer_states.size()[2]:self.buffer_states.size()[2]+self.buffer_actions.size()[2]]
        self.buffer_rewards = unique_tau[:,:,-1].unsqueeze(2)

        """ REMOVE ANY IMPORTANCE WEIGHTS EQUAL TO ZERO NUMERICALLY """
        while True:

            """ REMOVE THE IMPORTANCE WEIGHTS EQUAL TO ZERO NUMERICALLY """
            # self.buffer_iw, _ = self.compute_buffer_iw(policy, self.buffer_states, self.buffer_actions, self.buffer_rewards)
            self.buffer_iw = self.chosen_buffer(self.buffer_rewards)
            non_zero_indices = torch.nonzero(self.buffer_iw > 1e-12).reshape(-1)
            self.buffer_states = self.buffer_states [non_zero_indices,:,:]
            self.buffer_actions = self.buffer_actions[non_zero_indices,:,:]
            self.buffer_rewards = self.buffer_rewards[non_zero_indices,:,:]

            """ REAPPLY SELF NORMALIZATION TO SET AS DISCRETE DISTRIBUTION """
            if len(non_zero_indices) > 1:
                # self.buffer_iw, _ = self.compute_buffer_iw(policy, self.buffer_states, self.buffer_actions, self.buffer_rewards)
                self.buffer_iw = self.chosen_buffer(self.buffer_rewards)
            else:
                self.buffer_iw = torch.tensor([1.])
            # set the current buffer size
            self.current_buffer_size = len(self.buffer_iw)
            # if all the importance weights are greater then 0 the break
            if torch.eq(torch.sum(self.buffer_iw > 1e-12),  torch.tensor(self.buffer_iw.size()[0])):
                break

        """ MAKE THE IW ACTUALLY SUM TO ONE """
        self.buffer_iw = self.buffer_iw / torch.sum(self.buffer_iw)

        # nothing to return
        return None

    """ MAIN FUNCTION TO UPDATE BUFFER (GREEDY)"""
    def update_buffer(self, state_tensor, action_tensor, reward_tensor):

        """ MAIN UPDATE OF BUFFER """
        # check that its initialized
        if not self.buffer_initialized:

            """ FIND THE K LARGEST IW """
            # compute the importance weights of the current distribution
            iw = self.chosen_buffer(reward_tensor)
            # sort based on importance weights
            s, idx = iw.sort(descending=True)
            # only take the up to the number of values possible to store in buffer
            samples = idx[0:min(self.buffer_size, len(idx))]
            # store these samples
            self.buffer_states = state_tensor[samples,:,:].float()
            self.buffer_actions = action_tensor[samples,:,:].float()
            self.buffer_rewards = reward_tensor[samples,:,:].float()

            """ NOW CLEAN THE BUFFER GENERATED AND DEFINE IMPORTANCE WEIGHTS """
            self.clean_buffer()
            self.buffer_initialized = True

        else:
            # combine the state action stuff
            state_tensor = torch.cat([state_tensor.float(), self.buffer_states], dim=0)
            action_tensor = torch.cat([action_tensor.float(), self.buffer_actions], dim=0)
            reward_tensor = torch.cat([reward_tensor.float(), self.buffer_rewards], dim=0)
            # compute the importance weights of the current distribution
            iw = self.chosen_buffer(reward_tensor,resample_flag=True)
            # sort based on importance weights
            s, idx = iw.sort(descending=True)
            # only take the up to the number of values possible to store in buffer
            samples = idx[0:min(self.buffer_size, len(idx))]
            # store these samples
            self.buffer_states = state_tensor[samples,:,:]
            self.buffer_actions = action_tensor[samples,:,:]
            self.buffer_rewards = reward_tensor[samples,:,:]

            """ NOW CLEAN THE BUFFER GENERATED AND DEFINE IMPORTANCE WEIGHTS """
            self.clean_buffer()

        """ NOTHING TO RETURN """
        return None
