"""
GridWorld game implementation.
"""

import random
import numpy as np


class GridWorld:
    
    def __init__(self, env_y=8, env_x=8, init_agent_pos=None, goal_pos=None, max_steps=200, 
                 transition_probability=1.0):
        # Check gridworld setup
        assert (env_y > -1 and env_x > -1 and 
                isinstance(env_y, int) and isinstance(env_x, int)), 'GridWorld dimensions must be positive integers'
        if init_agent_pos:
            assertion_warning = 'Agent initial position must be within grid'
            assert (init_agent_pos[0] > -1 and init_agent_pos[0] < env_y), assertion_warning
            assert (init_agent_pos[1] > -1 and init_agent_pos[1] < env_x), assertion_warning
        if goal_pos:
            assertion_warning = 'Goal position must be within grid'
            assert (goal_pos[0] > -1 and goal_pos[0] < env_y), assertion_warning
            assert (goal_pos[1] > -1 and goal_pos[1] < env_x), assertion_warning
        
        # State space
        self.env_y = env_y
        self.env_x = env_x
        self.state_dim = (env_y, env_x)
        
        # Action space
        self.action_dim = (4,)  # up, right, down, left
        self.action_dict = {"up": 0, "right": 1, "down": 2, "left": 3}
        self.action_coords = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # translations
        
        # Initial state and observation
        self.__init_agent_pos = init_agent_pos
        self.__agent_pos = self.__get_random_pos() if init_agent_pos is None else init_agent_pos
        self.__goal_pos = self.__get_random_pos() if goal_pos is None else goal_pos
        self.obs = self.__build_state(self.__agent_pos, self.__goal_pos)
        
        # Steps
        self.__max_steps = max_steps
        self.__step = 0
        
        # Transition model
        self.__transition_probability = transition_probability

    def reset(self):
        # Reset setup
        self.__agent_pos = self.__get_random_pos() if self.__init_agent_pos is None else self.__init_agent_pos
        self.obs = self.__build_state(self.__agent_pos, self.__goal_pos)
        self.__step = 0
        return self.obs

    def step(self, action):
        self.__step += 1
        # Evolve agent state and build next observation
        if random.random() > self.__transition_probability:
            # Choose a random action for probabilistic transition functions
            available_actions = list(self.action_dict.items())
            actions_choice = list(filter(lambda a: a != int(action), available_actions))
            _, action = random.choice(actions_choice)
        self.__agent_pos = self.__make_move(action)
        self.obs = self.__build_state(self.__agent_pos, self.__goal_pos)
        # Collect reward
        reward = self.__build_reward()
        # Terminate if agent reached goal position
        done = self.__game_over()
        return self.obs, reward, done
    
    def render(self):
        board = np.array([[' '] * self.env_x] * self.env_y, dtype='U')
        if not self.__goal_reached():
            board[self.__goal_pos[0], self.__goal_pos[1]] = 'G' # goal
        board[self.__agent_pos[0], self.__agent_pos[1]] = 'A' # agent
        self.__plot_matrix(board)
    
    def __build_state(self, agent_pos, goal_pos):
        return tuple(np.array([state_var for state_var in (agent_pos + goal_pos)]))
    
    def __make_move(self, action):
        if action not in self.__get_allowed_actions():
            return self.__agent_pos
        agent_next_state = (self.obs[0] + self.action_coords[action][0],
                            self.obs[1] + self.action_coords[action][1])
        return agent_next_state
    
    def __get_allowed_actions(self):
        # Generate list of actions allowed depending on agent grid location
        actions_allowed = []
        y, x = self.obs[0], self.obs[1]
        if y > 0: # no passing top-boundary
            actions_allowed.append(self.action_dict["up"])
        if y < self.env_y - 1: # no passing bottom-boundary
            actions_allowed.append(self.action_dict["down"])
        if x > 0: # no passing left-boundary
            actions_allowed.append(self.action_dict["left"])
        if x < self.env_x - 1: # no passing right-boundary
            actions_allowed.append(self.action_dict["right"])
        actions_allowed = np.array(actions_allowed, dtype=int)
        return actions_allowed
    
    def __goal_reached(self):
        return self.__agent_pos == self.__goal_pos
    
    def __game_over(self):
        return self.__step >= self.__max_steps or self.__goal_reached()

    def __build_reward(self):
        return 0 if self.__goal_reached() else -1
    
    def __get_random_pos(self):
        return (random.randint(0, self.env_y - 1), random.randint(0, self.env_x - 1))
    
    def __plot_matrix(self, data):
        print(' –' * self.env_y)
        for row in data:
            print('|', end='')
            print(''.join(['{}|'.format(item) for item in row]))
        print(' –' * self.env_y)
