import gym
import numpy as np
import torch
from policy_network import PolicyNetwork
from torch.distributions import Bernoulli
from torch.autograd import Variable
from torch.distributions import Categorical

# code for acting and learning mostly obtained from: 
# https://github.com/pytorch/ignite/blob/master/examples/reinforcement_learning/reinforce.py

# policy gradient learning agent
class Agent():
    def __init__(self):
        self.lr = 1e-4
        self.decay_rate = 0.99
        self.gamma = 0.99
        self.policy = PolicyNetwork(learning_rate=self.lr, decay_rate=self.decay_rate)

    # select action based on state
    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        action_probs = self.policy(Variable(state))

        # use output probabilities of policy network and sample distribution
        m = Categorical(action_probs)
        action = m.sample()
        self.policy.saved_log_probs.append(m.log_prob(action))
        return action.data[0]

    def learn(self):
        R = 0
        policy_loss = []
        rewards = []
        for r in self.policy.rewards[::-1]:
            R = r + self.gamma * R
            rewards.insert(0, R)
            
        # turn rewards to pytorch tensor and standardize
        rewards = torch.Tensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)

        for log_prob, reward in zip(self.policy.saved_log_probs, rewards):
            policy_loss.append(-log_prob * reward)

        self.policy.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.policy.optimizer.step()

        del self.policy.rewards[:]
        del self.policy.saved_log_probs[:]


    # preprocessing method obtained from Karpathy's blog 
    def preprocess(self, state):
        # crop and flatten observation (numbers obtained from Andrej's code)
        state = state[35:195]

        # downsample observation by a factor of 2
        # :: syntax selects every second value along that dimension
        state = state[::2, ::2, :]

        # convert array to grayscale
        state = state[:, :, 0]

        # remove background from image
        state[state == 144] = 0
        state[state == 109] = 0

        state[state != 0] = 1

        # flatten to 6400 x 1 array of floats
        state = state.astype(np.float).ravel()

        return state

    def save_weights(self):
        torch.save(self.policy.state_dict(), 'pg_params.pkl')
        print("Weights saved.")

    def load_weights(self):
        self.policy.load_state_dict(torch.load('pg_params.pkl', map_location=lambda storage, loc: storage))
        print("Weights loaded.")
