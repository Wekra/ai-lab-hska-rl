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
        self.Q = np.zeros((*state_dim,*state_dim,self.action_size)) #Q(state_y, state_x, target_y, target_x, action) = Q-Value


    def act(self, state: Tuple[int, int, int, int]) -> int:
        """Selects the action to be executed based on the given state.
        
        Implements epsilon greedy exploration strategy, i.e. with a probability of
        epsilon, a random action is selected.
        
        Args:
            state: Tuple of agent and target position, representing the state.
        
        Returns:
            Action.
        """
        
        if random.random() <= self.epsilon:
            bestAction = random.randint(0,self.action_size - 1)
        else:
            Qrow = self.Q[state]
            #print(Qrow)
            bestAction = np.argmax(Qrow)
        return bestAction

    
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
        
        max_q_delta = np.max(self.Q[next_state])-current_Q_value
                
        
        self.Q[current_state][action] = current_Q_value + self.alpha * (reward + (self.gamma * max_q_delta))
        
        
    