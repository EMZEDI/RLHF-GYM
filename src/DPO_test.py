import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import math
from tqdm import tqdm

from DPO import PolicyModel

import gym
import matplotlib.pyplot as plt
from rewardmodelsimulator import RewardModelSimulator

policy_model = PolicyModel()

# Load the saved state dictionary
state_dict = torch.load('models/dpo_reference_model.pth')

# Load the state dictionary into the model
policy_model.load_state_dict(state_dict)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_model.to(device)

# Set the model to evaluation mode
policy_model.eval()

for x in range (30):
    for y in range (30):
        # Convert inputs to tensors
        state_x = torch.tensor([x], device=device, dtype=torch.float)
        state_y = torch.tensor([y], device=device, dtype=torch.float)

        output = torch.exp(policy_model(state_x, state_y)).tolist()

        print(f'Probs at state ({x}, {y}): {output[0]}')