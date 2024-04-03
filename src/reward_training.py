"""
Reward Training

This script trains a neural network model to predict the reward of a given state-action pair in a reinforcement learning environment.

Dependencies:
- torch: For creating and training the neural network model.
- torch.nn: For defining the neural network architecture.
- torch.optim: For defining the optimizer used in training.
- torch.utils.data: For creating data loaders.
- tqdm: For displaying a progress bar during training.

Data Preparation:
The script reads a CSV file containing state-action pairs and their associated rewards. Each line in the file represents a state-action pair and its reward. The script reads two lines at a time, treating them as a pair of state-action pairs. The data is split into a training set (95% of the data) and a validation set (5% of the data).

Model Definition:
The script defines a simple feed-forward neural network with one hidden layer. The network takes four inputs (the x and y coordinates of the state and the x and y components of the action) and outputs a single value representing the predicted reward.

Training Process:
The script trains the model for a specified number of epochs (100 in this case). During each epoch, the model's parameters are updated to minimize the binary cross-entropy loss between the predicted and actual rewards. The script also uses L1 regularization to prevent overfitting.

The training process includes a mechanism for early stopping. If the validation loss does not decrease for a specified number of consecutive epochs (5 in this case), the training process is stopped early.

The script calculates and prints the training and validation loss and accuracy after each epoch.

Saving:
The trained model is saved to a file named 'reward_model.pth' in the 'models' directory.

"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import math
from tqdm import tqdm

# Step 2: Model Definition
class RewardModel(nn.Module):
    def __init__(self):
        super(RewardModel, self).__init__()
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 1)

    def forward(self, state_x, state_y, action_x, action_y):
        # print(state_x.shape, state_y.shape, action_x.shape, action_y.shape)
        state_x = state_x.unsqueeze(1)
        state_y = state_y.unsqueeze(1)
        action_x = action_x.unsqueeze(1)
        action_y = action_y.unsqueeze(1)
        x = torch.cat((state_x, state_y, action_x, action_y), dim=1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    
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
    model = RewardModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.005)
    criterion = nn.BCEWithLogitsLoss()  # Binary cross-entropy loss
    regularization = nn.L1Loss()  # L1 regularization

    # Training loop
    best_valid_loss = float('inf')
    patience = 5  # Number of epochs to wait for improvement before early stopping
    counter = 0

    for epoch in range(100):  # Example: Train for 100 epochs
        model.train()
        train_loss = 0.0
        train_correct = 0
        for batch in tqdm(train_loader, desc=f'Epoch {epoch + 1}'):
            # print(batch.shape)
            optimizer.zero_grad()
            pair_a, pair_b = batch[:,0,:].to(device), batch[:,1,:].to(device)
            outputs_a = model(pair_a[:, 0], pair_a[:, 1], pair_a[:, 2], pair_a[:, 3])
            outputs_b = model(pair_b[:, 0], pair_b[:, 1], pair_b[:, 2], pair_b[:, 3])
            reward_a, reward_b = outputs_a.squeeze(), outputs_b.squeeze()

            # Normalize reward scores
            reward_a_norm = torch.sigmoid(reward_a)
            reward_b_norm = torch.sigmoid(reward_b)

            # Bradley-Terry probability
            p_ab = reward_a_norm / (reward_a_norm + reward_b_norm)

            # Calculate loss with regularization assuming that the first point has always the label 1
            loss = criterion(p_ab, pair_a[:, 4])
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

        # Validation
        model.eval()
        valid_loss = 0.0
        valid_correct = 0
        with torch.no_grad():
            for batch in valid_loader:
                pair_a, pair_b = batch[:,0,:].to(device), batch[:,1,:].to(device)
                outputs_a = model(pair_a[:, 0], pair_a[:, 1], pair_a[:, 2], pair_a[:, 3])
                outputs_b = model(pair_b[:, 0], pair_b[:, 1], pair_b[:, 2], pair_b[:, 3])
                reward_a, reward_b = outputs_a.squeeze(), outputs_b.squeeze()

                reward_a_norm = torch.sigmoid(reward_a)
                reward_b_norm = torch.sigmoid(reward_b)

                p_ab = reward_a_norm / (reward_a_norm + reward_b_norm)

                loss = criterion(p_ab, pair_a[:, 4])
                valid_loss += loss.item()

                # Calculate accuracy
                pred = (p_ab > 0.5).float()
                correct = (pred == pair_a[:, 4]).float().sum()
                valid_correct += correct

        valid_loss /= len(valid_loader)
        valid_accuracy = valid_correct / len(valid_loader.dataset)

        # Early stopping
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        print(f"Epoch {epoch + 1}: Train Loss = {train_loss:.4f}, Train Accuracy = {train_accuracy:.4f}, Valid Loss = {valid_loss:.4f}, Valid Accuracy = {valid_accuracy:.4f}")

    # Save trained model
    torch.save(model.state_dict(), 'models/reward_model.pth')
