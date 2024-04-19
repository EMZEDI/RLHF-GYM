from src.rewardmodelsimulator import RewardModelSimulator
import numpy as np

class RLHFEnv(RewardModelSimulator):
    def __init__(self, grid_size=30, distance='manhattan'):
        super().__init__(grid_size, distance)
        self.goal_state = np.array([29, 29])

    def step(self, action):
        # Calculate the new state based on the action
        new_state = self.state + self.translate(action)

        # Check if the new state is valid
        if not (0 <= new_state[0] < self.grid_size and 0 <= new_state[1] < self.grid_size):
            # If the new state is not valid, return the current state and a reward of -1
            return self.state, -1, False, {}

        # Update the state
        self.state = new_state

        # Check if the new state is the goal state
        if np.array_equal(self.state, self.goal_state):
            # If the new state is the goal state, return the new state and a reward of 200
            return self.state, 200, True, {}

        # If the new state is not the goal state, return the new state and a reward of -1
        return self.state, -1, False, {}