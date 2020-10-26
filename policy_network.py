import numpy as np
import os
import gym
import matplotlib.pyplot as plt
import retro
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
from torch.autograd import Variable
from torch.distributions import Categorical

# simple policy network used for policy gradient learning, from Karpathy's blog but output size changed to 3
class PolicyNetwork(nn.Module):
    def __init__(self, learning_rate, decay_rate):
        super(PolicyNetwork, self).__init__()

        self.fc1 = nn.Linear(6400, 200)

        # output space is 3 as you can do nothing, move upwards, or move downwards
        self.fc2 = nn.Linear(200, 3)

        self.optimizer = optim.RMSprop(self.parameters(), lr=learning_rate, weight_decay=decay_rate)

        self.saved_log_probs = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.softmax(self.fc2(x), dim=1)
        return x
