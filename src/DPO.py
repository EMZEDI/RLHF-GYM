import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import math
from tqdm import tqdm

# Policy Model Definition
class PolicyModel(nn.Module):
    def __init__(self):
        super(PolicyModel, self).__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 4) # 4 actions
    
    def forward(self, state_x, state_y):
        state_x = state_x.unsqueeze(1)
        state_y = state_y.unsqueeze(1)
        x = torch.cat((state_x, state_y), dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.log_softmax(self.fc2(x))
        return x

# DPO Loss Function, adapted from https://arxiv.org/abs/2305.18290
def dpo_loss(policy_logprobs, ref_logprobs, yw_indices, yl_indices, beta):
    """
    policy_logprobs: policy log probabilities, shape (B,)
    ref_logprobs: reference model log probabilities, shape (B,)
    yw_indices: preferred completion indices in [0, B-1], shape (T,)
    yl_indices: dispreferred completion indices in [0, B-1], shape (T,)
    beta: temperature controlling strength of KL penalty
    Each pair of (yw_indices[i], yl_indices[i]) represents the
    indices of a single preference pair.
    """

    # Extract Policy and Reference Log-Probabilities
    policy_yw_logprobs, policy_yl_logprobs = policy_logprobs[yw_indices], policy_logprobs[yl_indices]
    ref_yw_logprobs, ref_yl_logprobs = ref_logprobs[yw_indices], ref_logprobs[yl_indices]

    # Compute Log-Ratios for Policy and Reference Model
    policy_logratios = policy_yw_logprobs - policy_yl_logprobs
    ref_logratios = ref_yw_logprobs - ref_yl_logprobs

    # Calculate Losses
    losses = -nn.functional.logsigmoid(beta * (policy_logratios - ref_logratios))
    return losses

# Reference Model
up_prob = 0.4
down_prob = 0.25
left_prob = 0.25
right_prob = 0.1

# Create a tensor to store the reference log-probabilities
# Each row corresponds to a state-action pair
num_actions = 4  # Number of actions (cardinal directions)
num_states = 30 * 30  # Number of states
ref_logprobs = torch.zeros(num_states, num_actions)

# Set reference log-probabilities for movements in each cardinal direction
ref_logprobs[:, 0] = torch.log(torch.tensor(up_prob))  # Log-probability for up action
ref_logprobs[:, 1] = torch.log(torch.tensor(down_prob))  # Log-probability for down action
ref_logprobs[:, 2] = torch.log(torch.tensor(left_prob))  # Log-probability for left action
ref_logprobs[:, 3] = torch.log(torch.tensor(right_prob))  # Log-probability for right action

print("Reference Log-Probabilities:")

if __name__ == '__main__':

    if (torch.cuda.is_available()):
        device = torch.device('cuda')
    elif (torch.backends.mps.is_available()):
        device = torch.device('mps')
    else:
        device = torch.device('cpu')

    print(f'Using device: {device}')

    # Step 1: Data Preparation
    # Load the synthetic dataset
    data = []
    with open('dataset/minigrid_RLHF_dataset.csv', 'r') as f:
        next(f)  # Skip header
        state = None
        for line1, line2 in zip(f, f):  # Read two lines at a time
            parts1 = line1.strip().split(',')
            parts2 = line2.strip().split(',')
            state_x1, state_y1, action_x1, action_y1, preference1 = map(float, parts1)
            
            # encode action into scalar
            if action_x1 == 0:
                action_1 = (action_y1 - 1) // -2
            else:
                action_1 = 2 + (action_x1 - 1) // -2
            
            state_x2, state_y2, action_x2, action_y2, preference2 = map(float, parts2)
            data.append([(state_x1, state_y1, action_x1, action_y1, preference1),
                        (state_x2, state_y2, action_x2, action_y2, preference2)])

    # Convert data to PyTorch tensors
    data = [torch.tensor(pair, dtype=torch.float32) for pair in data]

    # Split data into training and validation sets
    split = int(0.95 * len(data))  # Use 95% of data for training, 5% for validation
    train_data = data[:split]
    valid_data = data[split:]

    # Create DataLoader for training and validation
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=32, shuffle=False)


    # Step 3: Training Process
    # Initialize model, optimizer, loss function, and regularization
    policy_model = PolicyModel().to(device)
    optimizer = optim.Adam(policy_model.parameters(), lr=0.01)
    regularization = nn.L1Loss()  # L1 regularization

    # Training loop
    best_valid_loss = float('inf')
    patience = 5  # Number of epochs to wait for improvement before early stopping
    counter = 0

    for epoch in range(100):  # Example: Train for 100 epochs
        # Training
        train_loss = 0.0
        train_correct = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}'):
            policy_logprobs = policy_model.forward()
            loss = dpo_loss()
            # print(batch.shape)
            optimizer.zero_grad()
            pair_a, pair_b = batch[:,0,:].to(device), batch[:,1,:].to(device)
            outputs_a = policy_model(pair_a[:, 0], pair_a[:, 1], pair_a[:, 2], pair_a[:, 3])
            outputs_b = policy_model(pair_b[:, 0], pair_b[:, 1], pair_b[:, 2], pair_b[:, 3])
            reward_a, reward_b = outputs_a.squeeze(), outputs_b.squeeze()

            # Normalize reward scores
            reward_a_norm = torch.sigmoid(reward_a)
            reward_b_norm = torch.sigmoid(reward_b)

            # Bradley-Terry probability
            p_ab = reward_a_norm / (reward_a_norm + reward_b_norm)

            # Calculate loss with regularization assuming that the first point has always the label 1
            loss = dpo_loss(policy_logprobs, ref_logprobs, rewards)
            loss += 0.001 * torch.sum(torch.abs(reward_a))  # L1 regularization
            loss += 0.001 * torch.sum(torch.abs(reward_b))  # L1 regularization
            loss += 0.001 * torch.sum(torch.abs(model.fc1.weight))  # L1 regularization
            loss += 0.001 * torch.sum(torch.abs(model.fc2.weight))  # L1 regularization

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

            # Calculate accuracy
            pred = (p_ab > 0.5).float()
            correct = (pred == pair_a[:, 4]).float().sum()
            train_correct += correct
        
        train_loss /= len(train_loader)
        train_accuracy = train_correct / len(train_loader.dataset)