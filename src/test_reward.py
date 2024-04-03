# A script to test the reward model

# load the torch model 
import torch
from src.reward_training import RewardModel

model = RewardModel()

model.load_state_dict(torch.load('models/reward_model.pth'))

# Set the model to evaluation mode
model.eval()

# Create a tensor for a single state-action pair
test = torch.tensor([[23.0, 26.0, 0.0, 1.0]])

# Run the model
output = model(test[0][0].unsqueeze(0), test[0][1].unsqueeze(0), test[0][2].unsqueeze(0), test[0][3].unsqueeze(0))

print(output)
