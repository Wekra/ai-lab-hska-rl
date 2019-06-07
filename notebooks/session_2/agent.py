"""
Q-Learning implementation.
"""

import random
import math

import numpy as np
from typing import Tuple

from lib import AbstractAgent
            
class AdvancedQLearning(AbstractAgent):

    def __init__(self, action_size: Tuple[int], buckets: Tuple[int, int, int, int],
                 gamma: float = None, epsilon: float = None, epsilon_min: float = None, 
                 alpha: float = None, alpha_min: float = None):
        self.action_size = action_size

        self.gamma = gamma  # discount factor (how much discount future reward)
        self.epsilon = epsilon  # exploration rate for the agent
        self.alpha = alpha  # learning rate
        
        self.epsilon_min = epsilon_min
        self.alpha_min = alpha_min
        
        self.epsilon_start = epsilon
        self.alpha_start = alpha
        
        # Initialize Q[s,a] table
        self.Q = np.zeros(buckets + (self.action_size,), dtype=float)
#         print(self.Q)
        self.t = 0 # played episodes

    def act(self, state: Tuple[int, int, int, int]) -> int:
        """Selects the action to be executed based on the given state.
        
        Implements epsilon greedy exploration strategy, i.e. with a probability of
        epsilon, a random action is selected.
        
        Args:
            state: Tuple of agent and target position, representing the state.
        
        Returns:
            Action.
        """
        if np.random.rand() <= self.epsilon: #do exploration
#             return random.randint(0, self.action_size-1)
            return random.randrange(self.action_size)
    
        return np.argmax(self.Q[state])

    def train(self, experience: Tuple[Tuple[int, int, int, int], int, Tuple[int, int, int, int], float, bool]) -> None:
        """Learns the Q-values based on experience.
        
        Args:
            experience: Tuple of state, action, next state, reward, done.
        
        Returns:
            None
        """
        current_state = experience[0]
        action = experience[1]
        next_state = experience[2]
        reward = experience[3]
        
        current_Q_value = self.Q[current_state][action]
        #print('current q ', current_Q_value,' at ', current_state)
        max_q_delta = np.max(self.Q[next_state])-current_Q_value
        
        self.Q[current_state][action] = current_Q_value + self.alpha * (reward + (self.gamma * max_q_delta))

    
        if experience[4]:
            self.epsilon = max(self.epsilon_min, min(self.epsilon_start, 1.0 - math.log10((self.t + 1) / 25)))
            self.alpha = max(self.alpha_min, min(self.alpha_start, 1.0 - math.log10((self.t + 1) / 25)))

#         self.epsilon = np.argmax([(9-np.log(self.t))/(1000*np.e), self.epsilon_min])
#         self.alpha = np.argmax([(8.4-np.log(0.65*self.t))/np.e, self.alpha_min])
            self.t += 1
      