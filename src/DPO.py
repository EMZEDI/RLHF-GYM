import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import math
from tqdm import tqdm

ACTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]

def read_preference_data():
    data = []
    with open('dataset/minigrid_RLHF_dataset.csv', 'r') as f:
        next(f)  # Skip header
        state = None
        for line1, line2 in zip(f, f):  # Read two lines at a time
            parts1 = line1.strip().split(',')
            parts2 = line2.strip().split(',')

            # Preference not used (reward modelling implicit in DPO)
            state_x1, state_y1, action_x1, action_y1, _ = map(float, parts1)
            state_x2, state_y2, action_x2, action_y2, _ = map(float, parts2)

            # Translate actions into scalars
            action_1 = ACTIONS.index((action_x1, action_y1))
            action_2 = ACTIONS.index((action_x2, action_y2))

            # Pair of preferred and dispreferred state-action pairs
            data.append([(state_x1, state_y1, action_1),
                        (state_x2, state_y2, action_2)])
    return data

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
        x = torch.log_softmax(self.fc2(x), dim=1)
        return x

# DPO Loss Function, adapted from https://arxiv.org/abs/2305.18290
def dpo_loss(policy_logprobs, ref_logprobs, yw_action_indices, yl_action_indices, beta):
    """
    policy_logprobs: policy log probabilities, shape (B, A)
    ref_logprobs: reference model log probabilities, shape (B, A)
    yw_action_indices: preferred completion indices in [0, B-1]
    yl_action_indices: dispreferred completion indices in [0, B-1]
    beta: temperature controlling strength of KL penalty
    Each pair of (yw_action_indices[i], yl_action_indices[i]) represents the
    action indices of a single preference pair.
    """

    # Extract preferred and dispreferred action probabilities
    policy_yw_logprobs = policy_outputs[torch.arange(policy_outputs.size(0)), yw_action_indices]
    policy_yl_logprobs = policy_outputs[torch.arange(policy_outputs.size(0)), yl_action_indices]
    ref_yw_logprobs = ref_logprobs[torch.arange(ref_logprobs.size(0)), yw_action_indices]
    ref_yl_logprobs = ref_logprobs[torch.arange(ref_logprobs.size(0)), yl_action_indices]

    # Compute Log-Ratios for Policy Model
    policy_logratios = policy_yw_logprobs - policy_yl_logprobs
    ref_logratios = ref_yw_logprobs - ref_yl_logprobs

    # Calculate Losses
    losses = -nn.functional.logsigmoid(beta * (policy_logratios - ref_logratios))
    # Compute Rewards
    rewards = beta * nn.functional.kl_div(torch.log_softmax(policy_logprobs, dim=1), 
              torch.softmax(ref_logprobs, dim=1), reduction='none').sum(dim=1).detach()

    # print(f'losses: {losses}')
    # print(f'rewards: {rewards}')
    return losses, rewards

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

            # Preference not used (reward modelling implicit in DPO)
            state_x1, state_y1, action_x1, action_y1, _ = map(float, parts1)
            state_x2, state_y2, action_x2, action_y2, _ = map(float, parts2)

            # Translate actions into scalars
            action_1 = ACTIONS.index((action_x1, action_y1))
            action_2 = ACTIONS.index((action_x2, action_y2))

            # Pair of preferred and dispreferred state-action pairs
            data.append([(state_x1, state_y1, action_1),
                        (state_x2, state_y2, action_2)])

    # Convert data to PyTorch tensors
    data = [torch.tensor(pair, dtype=torch.float32) for pair in data]

    # Split data into training and validation sets
    split = int(0.95 * len(data))  # Use 95% of data for training, 5% for validation
    train_data = data[:split]
    valid_data = data[split:]

    # Create DataLoader for training and validation
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    valid_loader = DataLoader(valid_data, batch_size=64, shuffle=False)

    # Step 2: Reference Model (maximizes likelihood of preferred completions)
    """
    From the original paper:
    In practice, one would like to reuse preference datasets publicly available,
    rather than generating samples and gathering human preferences.
    Since the preference datasets are sampled using π_SFT, we initialize π_ref = π_SFT whenever available.
    However, when π_SFT is not available [as is the case in this environment],
    we initialize π_ref by maximizing likelihood of preferred completions (x, y_w),
    that is, π_ref = argmax_π E_x,y_w∼D [log π(y_w | x)]
    """
    ref_model = PolicyModel().to(device)

    # Save reference model for more efficient training
    if not os.path.exists('models/dpo_reference_model.pth'):
        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        ref_optimizer = optim.Adam(ref_model.parameters(), lr=0.001)

        # Extract preferred state-action pairs and convert list of tensors into a single tensor
        preferred_data = torch.stack([pair[0] for pair in train_data]).to(device)
        preferred_loader = torch.utils.data.DataLoader(preferred_data, batch_size=64, shuffle=True)

        # Training loop
        num_epochs = 100
        for epoch in range(num_epochs):
            for batch in preferred_loader:
                states = batch[:, :2]  # Extract states
                actions = batch[:, 2].long()  # Extract actions
                
                # Calculate log probabilities using the policy model
                with torch.no_grad():
                    log_probs = ref_model(states[:, 0], states[:, 1])
                    log_probs_a = log_probs[torch.arange(log_probs.size(0)), actions]
                
                # Forward pass
                outputs = ref_model(states[:, 0], states[:, 1])
                
                # Compute loss
                loss = criterion(outputs, actions)
                
                # Zero gradients, backward pass, and optimize
                ref_optimizer.zero_grad()
                loss.backward()
                ref_optimizer.step()

        print(f"Reference Model Trained")
        torch.save(ref_model.state_dict(), 'models/dpo_reference_model.pth')
    else:
        ref_model.load_state_dict(torch.load('models/dpo_reference_model.pth'))
        print(f"Reference Model Loaded")

    # Step 3: Training Process
    # Initialize model, optimizer, loss function, and regularization
    policy_model = PolicyModel().to(device)
    # Match hyperparameters to original paper as much as possible
    optimizer = optim.RMSprop(policy_model.parameters(), lr=1e-6)
    regularization = nn.L1Loss()  # L1 regularization
    beta = 1

    # Training loop
    best_valid_loss = float('inf')
    patience = 5  # Number of epochs to wait for improvement before early stopping
    counter = 0

    for epoch in range(100):  # Example: Train for 100 epochs
        # Training
        train_loss = 0.0
        train_correct = 0

        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}'):
            # Zero gradients
            optimizer.zero_grad()

            # batch.shape: torch.Size([64, 2, 3])

            # Forward pass (extract preferred and dispreferred state-action pairs)
            pair_a, pair_b = batch[:,0,:].to(device), batch[:,1,:].to(device)
            # note: pair_a and pair_b have the same state values at indices 0 and 1, but their actions (index 2) differ
            policy_outputs = policy_model(pair_a[:, 0], pair_a[:, 1])
            ref_outputs = ref_model(pair_a[:, 0], pair_a[:, 1])
            
            # Extract preferred and dispreferred action probabilities
            preferred_policy_logprobs = policy_outputs[torch.arange(policy_outputs.size(0)), pair_a[:, 2].long()]
            dispreferred_policy_logprobs = policy_outputs[torch.arange(policy_outputs.size(0)), pair_b[:, 2].long()]

            # Calculate loss (mean computed for scalar output, needed for backpropagation)
            losses, rewards = dpo_loss(policy_outputs, ref_outputs, pair_a[:, 2].long(), pair_b[:, 2].long(), beta)
            # Weight loss pairwise using rewards (KL divergence from reference policy)
            loss = torch.mean(losses * rewards)
            train_loss += loss.item()

            # Backward loss
            loss.backward()

            # Update weights
            optimizer.step()
            
            # print('batch')
            # print(batch)
            # print('batch[:,0,:]')
            # print(pair_a)
            # print('batch[:,1,:]')
            # print(pair_b)
            # print(f'preferred preferred_policy_logprobs: {preferred_policy_logprobs}')
            # print(f'dispreferred_policy_logprobs: {dispreferred_policy_logprobs}')

            # Calculate accuracy
            correct = (preferred_policy_logprobs > dispreferred_policy_logprobs).float().sum().item()
            train_correct += correct
        
        print('epoch finished')
        
        train_loss /= len(train_loader)
        train_accuracy = train_correct / len(train_loader.dataset)

        print(f'train loss: {train_loss}')
        print(f'train accuracy: {train_accuracy}')
    
    # Save policy model
    torch.save(policy_model.state_dict(), 'models/dpo_policy_model.pth')