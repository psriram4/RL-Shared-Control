import numpy as np
import random
from agent import Agent

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


class DecisionRule():
    def __init__(self):
        self.aiAssistant =  Agent(state_size=STATE_SIZE, action_size=ACTION_SIZE, learning_rate=LEARNING_RATE,
                                learning_period=LEARNING_PERIOD, max_buffer_size=MAX_BUFFER_SIZE, discount=DISCOUNT,
                                tau=TAU, epsilon=EPSILON, epsilon_decay=EPSILON_DECAY, epsilon_end=EPSILON_END,
                                batch_size=BATCH_SIZE)

        self.aiAssistant.load_weights()
        self.aiAssistant.set_epsilon(EPSILON_END)

        self.agent_prob = 0.1

    def get_action(self, state, user_action):
            if random.random() < self.agent_prob:
                return self.aiAssistant.act(state)

            else:
                return user_action
