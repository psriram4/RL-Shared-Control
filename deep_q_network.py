import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import torchvision.transforms as T 


class DQN(nn.Module):
    def __init__(self, state_size, action_size, learning_rate, hidden_layer_dim):
        super(DQN, self).__init__()

        # paper uses multilayer perceptron with two hidden layers with 64 units each
        self.fc1 = nn.Linear(state_size, hidden_layer_dim)
        self.fc2 = nn.Linear(hidden_layer_dim, hidden_layer_dim)
        self.fc3 = nn.Linear(hidden_layer_dim, action_size)

        self.optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        self.loss = nn.MSELoss()

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x