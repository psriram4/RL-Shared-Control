import numpy as np
import random
from agent import Agent


class DecisionRule():
    def __init__(self):
        self.agent_prob = 0.1

    def get_action(self, state, user_action):
            # if random.random() < self.agent_prob:
            #     return self.aiAssistant.act(state)
            #
            # else:
            #     return user_action

        return user_action
