"""
synthetic-generation.py

This script generates a synthetic dataset for a reinforcement learning task in a grid world environment. 

The environment is defined by the MinigridEnv class from the src.environment module. The grid world is a square grid of a specified size, with a goal state and a set of possible actions that the agent can take at each state.

The script generates pairs of states and actions, along with the associated rewards. The rewards are calculated based on the Manhattan distance from the goal state, with some added randomness to simulate exploration.

The synthetic dataset is saved in a CSV file, with each row representing a state-action-reward tuple. The state and action are represented as 2D coordinates (x, y), and the reward is a single integer.

The script uses the following parameters:
- grid_size: The size of the grid world.
- epsilon: The probability of choosing a random action instead of the one that minimizes the distance to the goal.
- num_samples: The number of state-action-reward tuples to generate.

Dependencies:
- numpy
- copy
- src.environment
"""

import numpy as np
from copy import deepcopy
from src.environment import MinigridEnv

# Define the dimensions of the environment
grid_size = 30  # Adjust this based on your environment size

# Define the epsilon noise probability
epsilon = 0.001

# Generate synthetic dataset
num_samples = 100000  # Number of samples to generate

dataset = []

# Sample pairs of actions and states and generate the dataset
for _ in range(num_samples):
    env = MinigridEnv(grid_size=grid_size, distance='manhattan')
    s = env.reset()  # Random start state
    # check the final state
    while np.array_equal(s, env.goal_state):
        s = env.reset()
    a1 = np.random.randint(4)  # Random action 1

    # Ensure a1 leads to a valid state
    while not env.is_valid_state(env.possible_next_state(s, a1)):
        a1 = np.random.randint(4)

    # Ensure a2 is not equal to a1 and leads to a valid state
    a2 = np.random.randint(4)
    while a2 == a1 or not env.is_valid_state(env.possible_next_state(s, a2)):
        a2 = np.random.randint(4)

    # Deep copy the environment to avoid modifying the original one
    env_copy = deepcopy(env)

    # Get the states and rewards for each action
    s_prime_a1, reward_a1, _, _ = env.step(a1)
    s_prime_a2, reward_a2, _, _ = env_copy.step(a2)

    # Apply epsilon noise universally
    if np.random.uniform() < epsilon:
        reward = np.random.randint(2)  # Randomly assign reward 0 or 1
    else:
        # Calculate the rewards based on the distance
        if reward_a1 < reward_a2:
            reward = 1  # Action a1 is closer to the goal
        elif reward_a1 > reward_a2:
            reward = 0  # Action a2 is closer to the goal
        else:
            # If distances are equal, choose the action leading to a state with a higher y-coordinate
            if s_prime_a1[1] > s_prime_a2[1]:
                reward = 1
            else:
                reward = 0
    
    # Append data to dataset
    dataset.append((s, np.array(env.ACTIONS[a1]), reward))
    dataset.append((s, np.array(env.ACTIONS[a2]), 1 - reward))  # Reward for a2 is opposite of reward for a1

# Save the dataset as tabular format
with open('dataset/minigrid_RLHF_dataset.csv', 'w') as f:
    f.write("state_x,state_y,action_x,action_y,reward\n")
    for data in dataset:
        state_str = ','.join(map(str, data[0]))
        action_str = ','.join(map(str, data[1]))
        f.write(f"{state_str},{action_str},{data[2]}\n")
