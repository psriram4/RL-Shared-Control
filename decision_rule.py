import numpy as np
import random
from agent import Agent


class DecisionRule():
    def __init__(self):
        self.agent_prob = 0.1

    def get_action(self, state, user_action):
        """Select action for environment based on observation and user action.

        This rule is used with the 'play' mode.

        Parameters:
            state: current environment observation
            user_action: action retrieved from key press by user, if -1 no valid key press
        """

        # TODO: implement decision rule

        if user_action != -1:
            return user_action
        
        return 0
