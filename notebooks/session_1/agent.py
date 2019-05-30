"""
Q-Learning implementation.
"""

import random

import numpy as np
from typing import Tuple

from lib import AbstractAgent

class QLearning(AbstractAgent):

    def __init__(self, action_dim: Tuple, state_dim: Tuple,
                 gamma: float, epsilon: float, alpha: float):
        self.action_size = action_dim[0]

        self.gamma = gamma  # discount factor (how much discount future reward)
        self.epsilon = epsilon  # exploration rate for the agent
        self.alpha = alpha  # learning rate

        # Initialize Q[s,a] table
        # TODO
        self.Q = None

    def act(self, state: Tuple[int, int, int, int]) -> int:
        """Selects the action to be executed based on the given state.
        
        Implements epsilon greedy exploration strategy, i.e. with a probability of
        epsilon, a random action is selected.
        
        Args:
            state: Tuple of agent and target position, representing the state.
        
        Returns:
            Action.
        """
        # TODO
        return 0

    def train(self, experience: Tuple[Tuple[int, int, int, int], int, Tuple[int, int, int, int], float, bool]) -> None:
        """Learns the Q-values based on experience.
        
        Args:
            experience: Tuple of state, action, next state, reward, done.
        
        Returns:
            None
        """
        # TODO