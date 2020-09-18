import numpy as np
import gym
import random
from rl_agent import Agent
from imitation_agent import ImitationAgent
from collections import deque

NUM_EPISODES = 2000
MAX_STEPS = 1000
STATE_SIZE = 8
ACTION_SIZE = 4
LEARNING_RATE = 5e-4
LEARNING_PERIOD = 4
MAX_BUFFER_SIZE = 100000
DISCOUNT = 0.999
TAU = 1e-3
EPSILON = 1.0
EPSILON_DECAY = 0.995
EPSILON_END = 0.01
BATCH_SIZE = 64
HIDDEN_LAYER_DIM = 64
LOAD_WEIGHTS = True

class Dagger():
    def __init__(self):
        self.obs_dataset = []
        self.action_dataset = []
        self.expert = Agent(state_size=STATE_SIZE, action_size=ACTION_SIZE, learning_rate=LEARNING_RATE,
                                learning_period=LEARNING_PERIOD, max_buffer_size=MAX_BUFFER_SIZE, discount=DISCOUNT,
                                tau=TAU, epsilon=EPSILON, epsilon_decay=EPSILON_DECAY, epsilon_end=EPSILON_END,
                                batch_size=BATCH_SIZE)

        self.expert.load_weights()
        self.expert.set_epsilon(EPSILON_END)

        self.policy = ImitationAgent(state_size=STATE_SIZE, action_size=ACTION_SIZE, learning_rate=LEARNING_RATE,
                                learning_period=LEARNING_PERIOD, max_buffer_size=MAX_BUFFER_SIZE, discount=DISCOUNT,
                                tau=TAU, epsilon=EPSILON, epsilon_decay=EPSILON_DECAY, epsilon_end=EPSILON_END,
                                batch_size=BATCH_SIZE)

        self.env = gym.make('LunarLander-v2')


    def get_expert_action(self, state):
        return self.expert.act(state)

    def train(self, num_dagger_iter):
        recent_scores = deque(maxlen=150)
        episode_num = 0
        for i in range(num_dagger_iter):
            observations = []
            actions = []

            state = self.env.reset()
            episode_score = 0
            done = False

            for j in range(MAX_STEPS):
                if done:
                    break

                else:
                    observations.append(state)

                chosen_action = self.policy.act(state)
                actions.append(chosen_action)

                next_state, reward, done, info = self.env.step(chosen_action)

                # next_state, reward, done, info = self.env.step(self.get_expert_action(state))

                episode_score += reward
                state = next_state

            recent_scores.append(episode_score)

            if episode_num % 20 == 0:
                print("Episode: ", episode_num, ", Average Episode score: ", np.mean(recent_scores))
                
            episode_num += 1

            for obs in observations:
                self.obs_dataset.append(obs)
                self.action_dataset.append(self.get_expert_action(obs))

            self.policy.learn(self.obs_dataset, self.action_dataset)

        self.policy.save_weights()
