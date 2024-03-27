import gymnasium as gym
import minigrid
import pygame

class RLHFMinigrid(gym.Wrapper):
  def __init__(self, env):
    super().__init__(env)  # Call parent class constructor (gym.Wrapper)
    self.state = None  # Store the current state for feedback
    self.action_space = env.action_space  # Access the original action space

  def step(self, action):
    # Perform the action in the environment
    obs, reward, terminated, truncated, info, done = self.env.step(action)
    self.state = obs  # Update the current state after the action
    return obs, reward, terminated, truncated, info, done

  def reset(self):
    obs, info = self.env.reset()
    self.state = obs
    return obs, info

  def get_feedback(self, possible_actions: list[int, int]):
    # Convert the state dictionary (self.state) to a visual representation
    grid = self._render_grid(self.state)  # Call a helper function to render the grid

    # Get all possible actions from the action space
    num_actions = self.action_space.n  # Number of possible actions

    # Display the visual representation and wait for user click
    chosen_action_index = self._handle_user_click(grid, num_actions, possible_actions)
    return chosen_action_index
  
  import pygame
  
  def _render_grid(self, state):
    return screen   # render the screen and wait for the choice of the best action by the user
  
  def _handle_user_click(grid, num_actions, possible_actions: list[int, int]):
    return index # Return the index of the chosen action
  

