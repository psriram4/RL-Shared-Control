import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


class ImitationNetwork(nn.Module):
    def __init__(self, state_size, action_size, learning_rate):
        super(ImitationNetwork, self).__init__()

        self.state_size = state_size
        self.action_size = action_size

        # proposed dimensions for dense layers
        fc1_layer_dim = 16
        fc2_layer_dim = 16
        fc3_layer_dim = 16

        self.fc1 = nn.Linear(state_size, fc1_layer_dim)
        self.fc2 = nn.Linear(fc1_layer_dim, fc2_layer_dim)
        self.fc3 = nn.Linear(fc2_layer_dim, fc3_layer_dim)
        self.fc4 = nn.Linear(fc3_layer_dim, action_size)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss = nn.CrossEntropyLoss()


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.softmax(self.fc4(x))

        return x
