import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

# Implement model for learning as needed
class DuelingDQN(nn.Module):
    def __init__(self, state_size, action_size, learning_rate):
        super(DuelingDQN, self).__init__()

        self.state_size = state_size
        self.action_size = action_size

        # proposed dimensions for dense layers
        fc1_layer_dim = 64
        fc2_layer_dim = 64
        fc3_layer_dim = 32

        self.fc1 = nn.Linear(state_size, fc1_layer_dim)
        self.fc2 = nn.Linear(fc1_layer_dim, fc2_layer_dim)

        # Dueling DQN separates into value and advantage

        # value
        self.fc3_v = nn.Linear(fc2_layer_dim, fc3_layer_dim)
        self.fc4_v = nn.Linear(fc3_layer_dim, 1)

        # advantage
        self.fc3_a = nn.Linear(fc2_layer_dim, fc3_layer_dim)
        self.fc4_a = nn.Linear(fc3_layer_dim, action_size)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        value = F.relu(self.fc3_v(x))
        value = self.fc4_v(value)

        advantage = F.relu(self.fc3_a(x))
        advantage = self.fc4_a(advantage)

        # combine value and advantage to generate q values for each action
        action_q_vals = value + advantage - advantage.mean(1).unsqueeze(1)

        return action_q_vals
