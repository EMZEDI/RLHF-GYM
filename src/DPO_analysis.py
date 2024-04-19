import torch
from DPO import PolicyModel, simulate_rewards
from RLHFenvironment import RLHFEnv

num_episodes = 500
max_iters_per_episode = 200
env = RLHFEnv()

# Comparing performance of optimal action selection (benchmark 1), reference model (benchmark 2) and trained DPO policy model 
ref_model = PolicyModel()
dpo_policy_model = PolicyModel()

# Load the saved state dictionaries
ref_state_dict = torch.load('../models/dpo_reference_model.pth')
dpo_state_dict = torch.load('../models/dpo_policy_model.pth')

# Load the state dictionaries into each model
ref_model.load_state_dict(ref_state_dict)
dpo_policy_model.load_state_dict(dpo_state_dict)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ref_model.to(device)
dpo_policy_model.to(device)

# Set the model to evaluation mode
ref_model.eval()
dpo_policy_model.eval()

# Simulate rewards for optimal action selection
b1_cumulative_rewards = []

for episode in range(num_episodes):
    # initialize environment
    state = env.reset()
    cumulative_reward = 0

    for i in range(max_iters_per_episode):
        # find optimal action (move up until final row, then move right)
        action = 2 if state[1] == 29 else 0

        # convert state to tensor format
        state_x = torch.tensor([state[0]], device=device, dtype=torch.float)
        state_y = torch.tensor([state[1]], device=device, dtype=torch.float)

        state, reward, done, _ = env.step(action)

        cumulative_reward += reward
        if done: break

    b1_cumulative_rewards.append(cumulative_reward)

# Simulate rewards for reference policy
b2_cumulative_rewards = simulate_rewards(ref_model, env, device, num_episodes, max_iters_per_episode)
# Simulate rewards for DPO
dpo_cumulative_rewards = simulate_rewards(dpo_policy_model, env, device, num_episodes, max_iters_per_episode)

print(f'Optimal action selection results in mean reward {sum(b1_cumulative_rewards)/len(b1_cumulative_rewards)}')
print(f'Reference policy results in mean reward {b2_cumulative_rewards}')
print(f'DPO results in mean reward {dpo_cumulative_rewards}')