import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from model import DuelingDQN
from replay import ReplayMemory


class Agent():
    def __init__(self, num_episodes, learning_rate, max_steps, discount, batch_size):
        """Initialize agent."""

        # hyperparameters obtained from elsewhere
        self.state_size = 8
        self.action_size = 4

        self.tau = 1e-3

        self.learning_rate = learning_rate
        self.learning_period = 4

        self.max_buffer_size = 100000
        self.discount = discount
        
        self.epsilon = 1.0
        self.epsilon_decay = 0.995
        self.epsilon_end = 0.01

        self.batch_size = batch_size

        self.learning_steps = 0

        # initialize replay memory and two dueling q-networks for double q learning
        self.memory = ReplayMemory(self.max_buffer_size)
        self.primary_q_net = DuelingDQN(self.state_size, self.action_size, self.learning_rate)
        self.target_q_net = DuelingDQN(self.state_size, self.action_size, self.learning_rate)

    def act(self, observation):
        """Select agent action based on observation.

        Parameters:
            observation: current state of environment
        """

        # based on state, select agent action

        # explore with probability epsilon
        if random.random() < self.epsilon:
            action = random.randint(0, self.action_size - 1)
            return action

        state = torch.FloatTensor(observation).unsqueeze(0)

        self.primary_q_net.eval()

        with torch.no_grad():
            out = self.primary_q_net(state)

        # determine action based on policy
        optimal_action = torch.argmax(out, 1).item()

        return optimal_action

    def step(self, state, action, reward, next_state, done):
        """After each environment step, this agent step function will also execute. 

        Perform any learning and save transition to replay memory.
        
        Parameters:
            state: initial state
            action: action taken by agent
            reward: reward from action taken
            next_state: next state after action taken
            done: bool value whether episode is done
        """

        self.save_to_memory(state, action, reward, next_state, done)
        self.learn()
        self.learning_steps += 1

    def learn(self):
        if self.memory.buffer_size < self.batch_size:
            return

        # learn every n steps
        if self.learning_steps % self.learning_period != 0:
            return

        replay = self.memory.sample_memory(self.batch_size)
        batch_indices = np.arange(self.batch_size)

        # index all transition tuples in the batch
        states = [replay[i][0] for i in range(self.batch_size)]
        actions = [replay[i][1] for i in range(self.batch_size)]
        rewards = [replay[i][2] for i in range(self.batch_size)]
        next_states = [replay[i][3] for i in range(self.batch_size)]
        dones = [replay[i][4] for i in range(self.batch_size)]

        # convert to tensor
        states = torch.FloatTensor(states)
        next_states = torch.FloatTensor(next_states)
        actions = torch.tensor(actions, dtype=int).unsqueeze(-1)
        rewards = torch.FloatTensor(rewards).unsqueeze(-1)
        dones = torch.tensor(dones, dtype=int).unsqueeze(-1)
        batch_indices = torch.tensor(batch_indices, dtype=int)

        self.primary_q_net.eval()
        self.target_q_net.eval()

        primary_q_net_states = self.primary_q_net(states)
        target_q_net_next_states = self.target_q_net(next_states)
        primary_q_net_next_states = self.primary_q_net(next_states)

        actual_q = primary_q_net_states.gather(1, actions)
        max_next_state_actions = torch.argmax(primary_q_net_next_states, dim=1).unsqueeze(-1)

        self.primary_q_net.train()

        # calculate target q
        target_q = rewards + (self.discount * target_q_net_next_states.gather(1, max_next_state_actions) * (1 - dones))

        # backpropagation
        loss = self.primary_q_net.loss(target_q, actual_q)
        self.primary_q_net.optimizer.zero_grad()
        loss.backward()
        self.primary_q_net.optimizer.step()

        # update target network with primary network (double Q-learning)
        self.soft_update()

    def soft_update(self):
        # update code obtained from https://towardsdatascience.com/double-deep-q-networks-905dd8325412
        for target_param, param in zip(self.target_q_net.parameters(), self.primary_q_net.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)

    def set_epsilon(self, epsilon_value):
        self.epsilon = epsilon_value

    def reduce_epsilon(self):
        if (self.epsilon * self.epsilon_decay) >= self.epsilon_end:
            self.epsilon *= self.epsilon_decay

    def save_to_memory(self, state, action, reward, next_state, done):
        self.memory.save_transition(state, action, reward, next_state, done)

    def save_weights(self):
        torch.save(self.primary_q_net.state_dict(), 'primary_q_network_weights.pth')
        print("Weights saved.")

    def load_weights(self):
        self.primary_q_net.load_state_dict(torch.load('primary_q_network_weights.pth', map_location=lambda storage, loc: storage))
        print("Weights loaded.")
