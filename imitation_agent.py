import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from imitation_learning_net import ImitationNetwork


class ImitationAgent():
    def __init__(self, state_size, action_size, learning_rate, learning_period,
                max_buffer_size, discount, tau, epsilon, epsilon_decay, epsilon_end,
                batch_size):

        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.learning_period = learning_period

        self.max_buffer_size = max_buffer_size
        self.discount = discount
        self.tau = tau

        # set epsilon to 0.01 for testing purposes
        # self.epsilon = epsilon
        self.epsilon = 0.01

        self.epsilon_decay = epsilon_decay
        self.epsilon_end = epsilon_end
        self.batch_size = batch_size

        self.learning_steps = 0

        # initialize neural net for imitation learning
        self.model = ImitationNetwork(self.state_size, self.action_size, self.learning_rate)

    def act(self, observation):
        # explore with probability epsilon
        if random.random() < self.epsilon:
            action = random.randint(0, self.action_size - 1)
            return action

        state = torch.FloatTensor(observation).unsqueeze(0)

        self.model.eval()

        with torch.no_grad():
            out = self.model(state)

        # determine action based on policy
        optimal_action = torch.argmax(out, 1).item()

        return optimal_action

    def learn(self, observations, actions):
        policy_actions = []
        expert_actions = []

        self.model.eval()

        observations = torch.FloatTensor(observations)
        policy_actions = self.model(observations)

        # print("observations: ", observations)
        # print("expert policy actions: ", expert_policy_actions)
        #
        # for obs in observations:
        #     obs = torch.FloatTensor(obs).unsqueeze(0)
        #     policy_actions.append(self.model(obs))

        # print("actions: ", actions)

        for action in actions:
            expert_actions.append(action)
            # possible_actions = [0, 0, 0, 0]
            # possible_actions[action] = 1
            #
            # expert_actions.append(possible_actions)


        self.model.train()

        # print("before")
        # print(expert_actions)
        # print(policy_actions)

        expert_actions = torch.LongTensor(expert_actions)
        policy_actions = torch.FloatTensor(policy_actions)

        # print("after")
        # print(expert_actions)
        # print(policy_actions)

        # aggregated dataset - [....],


        # [0, 1, 0, 0] - expert_actions
        # [0.2, 0.4, 0.2, 0.2] - policy

        # loss = self.model.loss(expert_actions, policy_actions)
        loss = self.model.loss(policy_actions, expert_actions)
        self.model.optimizer.zero_grad()
        loss.backward()
        self.model.optimizer.step()


    def save_weights(self):
        torch.save(self.model.state_dict(), 'imitation_model.pth')
        print("Weights saved.")

    def load_weights(self):
        self.model.load_state_dict(torch.load('imitation_model.pth', map_location=lambda storage, loc: storage))
        print("Weights loaded.")
