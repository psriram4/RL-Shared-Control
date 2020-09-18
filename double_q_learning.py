import os
import numpy as np
import random
import gym

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F 
import torchvision.transforms as T 


class ExperienceReplay():
    def __init__(self, max_buffer_size):
        self.buffer_size = 0
        self.max_buffer_size = max_buffer_size
        self.memory_buffer = []

    def save_transition(self, state, action, reward, next_state, done):
        if self.buffer_size >= self.max_buffer_size:
            self.memory_buffer.pop(0)
            self.buffer_size -= 1
        
        transition = (state, action, reward, next_state, done)
        # print("transition: ")
        # print(transition)
        self.memory_buffer.append(transition)
        self.buffer_size += 1

    def sample_memory(self, batch_size):
        if batch_size > self.buffer_size:
            return

        memory_refs = np.arange(self.buffer_size)
        batch_indices = np.random.choice(memory_refs, batch_size)
        batch = [self.memory_buffer[i] for i in batch_indices]

        return batch


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
    
class Agent():
    def __init__(self, state_size, action_size, learning_rate, 
                max_buffer_size, discount, epsilon, batch_size,
                learn_period, hidden_layer_dim):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate
        self.max_buffer_size = max_buffer_size
        self.discount = discount
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.learn_period = learn_period
        self.learn_steps = 0

        # initialize replay memory and two q networks for double q learning
        self.memory = ExperienceReplay(self.max_buffer_size)
        self.primary_q_net = DQN(self.state_size, self.action_size, self.learning_rate, hidden_layer_dim)
        self.target_q_net = DQN(self.state_size, self.action_size, self.learning_rate, hidden_layer_dim)

    def act(self, observation):
        if random.random() < self.epsilon:
            action = random.randint(0, self.action_size - 1)
            return action

        state = torch.tensor(observation)
        out = self.primary_q_net.forward(state)
        optimal_action = torch.argmax(out).item()

        return optimal_action
    
    def learn(self):
        if self.memory.buffer_size < self.batch_size:
            return 

        if self.learn_steps % self.learn_period == 0:
            self.target_q_net.load_state_dict(self.primary_q_net.state_dict())

        batch_indices = np.arange(self.batch_size)
        
        replay = self.memory.sample_memory(self.batch_size)
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

        self.learn_steps += 1


    def save_transition(self, state, action, reward, next_state, done):
        self.memory.save_transition(state, action, reward, next_state, done)

        
env = gym.make('LunarLander-v2')
num_games = 500

agent = Agent(state_size=8, action_size=4, learning_rate=5e-4, max_buffer_size=100000, discount=0.99, epsilon=0.01, batch_size=64, learn_period=100, hidden_layer_dim=64)

scores, eps_history = [], []

print("starting...")
for i in range(num_games):
    done = False
    observation = env.reset()
    score = 0

    while not done:
        action = agent.act(observation)
        next_observation, reward, done, info = env.step(action)
        score += reward
        agent.save_transition(observation, action, reward, next_observation, int(done))
        # print("observation: ", observation)
        # print("action: ", action)
        # print("reward: ", reward)
        # print("next_observation: ", next_observation)
        # print("done: ", int(done))
        agent.learn()
        observation = next_observation

    scores.append(score)
    avg_score = np.mean(scores[-100:])
    print('episode ', i, 'score %.1f' % score, 'average score %.1f' % avg_score, 'epsilon %.2f' % agent.epsilon)
    eps_history.append(agent.epsilon)

