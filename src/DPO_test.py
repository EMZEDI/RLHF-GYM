import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import math
from tqdm import tqdm

from DPO import PolicyModel

# Define your PolicyModel
policy_model = PolicyModel()

# Load the saved state dictionary
state_dict = torch.load('models/dpo_policy_model.pth')

# Load the state dictionary into the model
policy_model.load_state_dict(state_dict)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
policy_model.to(device)

# Set the model to evaluation mode
policy_model.eval()

# Convert inputs to tensors
state_x = torch.tensor([0.0], device=device)
state_y = torch.tensor([0.0], device=device)

output = torch.exp(policy_model(state_x, state_y)).tolist()

print(output)