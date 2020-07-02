import numpy as np
import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import torchvision.transforms as T 

from deep_q_network import DQN
from experience_replay import ExperienceReplay 

class Agent():
    def __init__(self, state_size, action_size, learning_rate, learning_period,
                max_buffer_size, discount, tau, epsilon, epsilon_decay, epsilon_end,
                batch_size, hidden_layer_dim):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.learning_period = learning_period

        self.max_buffer_size = max_buffer_size
        self.discount = discount
        self.tau = tau
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_end = epsilon_end
        self.batch_size = batch_size
        self.hidden_layer_dim = hidden_layer_dim

        self.learning_steps = 0

        # initialize replay memory and two q networks for double q learning
        self.memory = ExperienceReplay(self.max_buffer_size)
        self.primary_q_net = DQN(self.state_size, self.action_size, self.learning_rate, hidden_layer_dim)
        self.target_q_net = DQN(self.state_size, self.action_size, self.learning_rate, hidden_layer_dim)

    def act(self, observation):
        if random.random() < self.epsilon:
            action = random.randint(0, self.action_size - 1)
            return action

        state = torch.tensor(observation)

        self.primary_q_net.eval()
        with torch.no_grad():
            out = self.primary_q_net(state)

        optimal_action = torch.argmax(out).item()
        return optimal_action
    
    def learn(self):
        if self.memory.buffer_size < self.batch_size:
            return 

        if self.learning_steps % self.learning_period != 0:
            # self.learning_steps += 1
            self.soft_update_target_network()
        

        replay = self.memory.sample_memory(self.batch_size)
        batch_indices = np.arange(self.batch_size)


        states = [replay[i][0] for i in range(self.batch_size)]
        actions = [replay[i][1] for i in range(self.batch_size)]
        rewards = [replay[i][2] for i in range(self.batch_size)]
        next_states = [replay[i][3] for i in range(self.batch_size)]
        dones = [replay[i][4] for i in range(self.batch_size)]

        states = torch.tensor(states)
        next_states = torch.tensor(next_states)
        actions = torch.tensor(actions)
        rewards = torch.tensor(rewards)
        dones = torch.tensor(dones)

        # self.primary_q_net.eval()
        # primary_states = self.primary_q_net(states)
        # primary_next_states = self.primary_q_net(next_states)

        # self.target_q_net.eval()
        # target_next_states = self.target_q_net(next_states)

        # max_next_state_actions = torch.argmax(primary_next_states, dim=1)
        
        # actual_q = []
        # target_predictions = []
        # for i in range(self.batch_size):
        #     if dones[i] == 1:
        #         actual_q.append(primary_states[i][actions[i]].item())
        #         target_predictions.append(0)
            
        #     else:
        #         actual_q.append(primary_states[i][actions[i]].item())
        #         target_predictions.append(target_next_states[i][max_next_state_actions[i]].item())

        # actual_q = torch.tensor(actual_q).requires_grad_()
        # target_predictions = torch.tensor(target_predictions).requires_grad_()

        # # print("actual q: ", actual_q)
        # # print("target_predictions: ", target_predictions)

        # target_q = rewards + (self.discount * target_predictions)

        # print("target_q: ", target_q)


        self.primary_q_net.eval()
        self.target_q_net.eval()
        primary_q_net_states = self.primary_q_net.forward(states)
        target_q_net_next_states = self.target_q_net.forward(next_states)
        primary_q_net_next_states = self.primary_q_net.forward(next_states)

        actual_q = primary_q_net_states[batch_indices, actions]
        max_next_state_actions = torch.argmax(primary_q_net_next_states, dim=1)

        # print("primary_q_net_states: ", primary_q_net_states)
        # for i in range(self.batch_size):
        #     print("chosen action: ", primary_q_net_states[i][actions[i]].item())
            
        # print("dones: ", done)
        target_q_net_next_states[dones] = 0.0
        target_q = rewards + (self.discount * target_q_net_next_states[batch_indices, max_next_state_actions])


        self.primary_q_net.train()
        loss = self.primary_q_net.loss(target_q, actual_q)
        self.primary_q_net.optimizer.zero_grad()
        loss.backward()
        self.primary_q_net.optimizer.step()

        self.learning_steps += 1

        if self.epsilon > (self.epsilon_end + self.epsilon_decay):
            self.epsilon -= self.epsilon_decay
            
        # self.soft_update_target_network()

    def soft_update_target_network(self):
        self.target_q_net.load_state_dict(self.primary_q_net.state_dict())
        
        # target_weights = target_weights * (1-TAU) + q_weights * TAU where 0 < TAU < 1
    #    for primary_q_net_param, target_q_net_param in zip(self.primary_q_net.parameters(), self.target_q_net.parameters()):
    #         target_q_net_param.data.copy_(self.tau * primary_q_net_param.data + (1.0 - self.tau) * target_q_net_param.data)


    def save_to_memory(self, state, action, reward, next_state, done):
        self.memory.save_transition(state, action, reward, next_state, done)
        
    def save_weights(self):
        torch.save(self.primary_q_net.state_dict(), "primary_q_network_weights.pth")
        torch.save(self.target_q_net.state_dict(), "target_q_network_weights.pth")

