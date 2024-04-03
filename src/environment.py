import gym
from gym import spaces
import numpy as np

class MinigridEnv(gym.Env):

    """
    MinigridEnv Class

    This class represents a grid world environment for a reinforcement learning task. It is a subclass of gym.Env, which is a base class for custom environments in OpenAI Gym.

    Methods:
    - __init__(self, grid_size=10, distance='manhattan'): Initializes the environment. The grid size and distance calculation method can be specified.

    - euclidean_distance(self, s, goal): Calculates the Euclidean distance between two states.

    - manhattan_distance(self, s, goal): Calculates the Manhattan distance between two states.

    - get_reward(self, s): Calculates the reward for a given state. The reward is the distance from the goal state.

    - step(self, action): Takes an action and returns the next state, reward, done flag, and info dictionary. The action is an index into the ACTIONS list.

    - reset(self): Resets the environment to an initial state, which is chosen randomly.

    - render(self, mode='human'): An optional method to render the environment. Not implemented in this version of the class.

    - close(self): An optional method to close any resources used by the environment. Not implemented in this version of the class.

    - possible_next_state(self, s, a): Returns the possible next state after taking an action at a given state.

    - is_valid_state(self, s): Checks if a state is valid (i.e., within the grid).

    Attributes:
    - ACTIONS: A list of possible actions, represented as movements in the grid.
    - action_space: The action space of the environment, represented as a Discrete space.
    - observation_space: The observation space of the environment, represented as a 2D Box space.
    - goal_state: The goal state of the environment.
    """

    # Define the action space
    ACTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Up, Down, Right, Left

    def __init__(self, grid_size=10, distance='manhattan'):
        super(MinigridEnv, self).__init__()
        self.grid_size = grid_size
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(4)  # Four discrete actions: 0, 1, 2, 3
        self.action_description = {0: 'Up', 1: 'Down', 2: 'Right', 3: 'Left'}
        self.observation_space = spaces.Box(low=0, high=grid_size-1, shape=(2,), dtype=np.int32)  # 2D box for observations
        
        # Define goal state
        self.goal_state = np.array((grid_size - 1, grid_size - 1))

        # Define the distance calculation formula
        if distance == 'manhattan':
            self.distance = self.manhattan_distance
        elif distance == 'euclidean':
            self.distance = self.euclidean_distance
        else:
            raise ValueError("Invalid distance calculation method")
        
        self.state = None   # should call reset first
        
    def euclidean_distance(self, s, goal):
        return np.linalg.norm(np.array(s) - np.array(goal))

    def manhattan_distance(self, s, goal):
        return abs(s[0] - goal[0]) + abs(s[1] - goal[1])

    def get_reward(self, s):
        distance = self.distance(s, self.goal_state)
        return distance

    def step(self, action):
        # Take an action and return the next state, reward, done, and info
        s_prime = (self.state[0] + self.ACTIONS[action][0], self.state[1] + self.ACTIONS[action][1])
        self.state = s_prime
        
        reward = self.get_reward(self.state)
        
        return np.array(self.state), reward, False, {}

    def reset(self):
        # Reset the environment to initial state
        self.state = (np.random.randint(self.grid_size), np.random.randint(self.grid_size))
        return np.array(self.state)

    def render(self, mode='human'):
        # Optional: render the environment
        pass

    def close(self):
        # Optional: close any resources used by the environment
        pass

    def possible_next_state(self, s, a):
        return (s[0] + self.ACTIONS[a][0], s[1] + self.ACTIONS[a][1])
    
    def is_valid_state(self, s):
        return 0 <= s[0] < self.grid_size and 0 <= s[1] < self.grid_size
