from environment import MinigridEnv
import torch
from src.reward_training import RewardModel
import numpy as np

class RLHFEnv(MinigridEnv):
    """
    This class provides an implementation of the RLHF (Reinforcement Learning with Human Feedback) environment. 
    The environment uses a pre-trained reward model to replace the traditional reward system based on RLHF principles.

    Actions:
        The environment accepts actions as integers from 0 to 3. These integers map to the following actions:
        - 0: Up
        - 1: Down
        - 2: Right
        - 3: Left
        These actions are represented internally as tuples, where each tuple corresponds to a change in the (x, y) position of the agent. 
        The mapping is as follows:
        ACTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]  # Up, Down, Right, Left

    Step Function:
        The `step` function is used to take an action in the environment. It accepts an integer from 0 to 3 as input, 
        which corresponds to the actions described above. The `step` function returns a tuple containing the following:
        - `state`: A 1D numpy array representing the current state of the environment.
        - `reward`: A float representing the reward for the taken action. The reward is determined by the pre-trained reward model.
        - `done`: A boolean indicating whether the episode has ended.
        - `info`: A dictionary that can be used to provide additional implementation-specific information. Its content is not specified and may vary between different implementations.

    Reset Function:
        The `reset` function is used to reset the environment to its initial state. It returns the initial state as a 1D numpy array.

    Reward Model:
        The environment uses a pre-trained reward model to determine the reward for each action. The model is trained based on RLHF principles, 
        which aim to align the agent's behavior with human values.
    """

    def __init__(self, grid_size=30, distance='manhattan'):
        super().__init__(grid_size, distance)
        # Add your custom initialization here

        self.model = RewardModel()
        # TODO: Could potentially use torch.device to use GPU if available
        self.model.load_state_dict(torch.load('models/reward_model.pth'))
        self.model.eval()

    def step(self, action):

        action_array = self.translate(action)
        # Assuming self.state and action are numpy arrays
        state_action = np.concatenate((self.state, action_array))
        state_action = torch.from_numpy(state_action).float().unsqueeze(0)

        reward = self.model(state_action[0][0].unsqueeze(0), state_action[0][1].unsqueeze(0), state_action[0][2].unsqueeze(0), state_action[0][3].unsqueeze(0))
        # only the scalar of the reward is needed
        reward = reward.item()

        # call the super class step method
        s_prime, reward_unused, done, info = super().step(action)

        if (np.array_equal(s_prime, self.goal_state)):
            done = True
        
        return s_prime, reward, done, info

    def translate(self, action):
        return np.array(self.ACTIONS[action])
