"""
This file contains the class for the PPO algorithm

Inspired by tutorial given by Eric Yang Yu: https://medium.com/analytics-vidhya/coding-ppo-from-scratch-with-pytorch-part-1-4-613dfc1b14c8
"""

import matplotlib.pyplot as plt

from evalPPO import eval_policy
from rewardmodelsimulator import RewardModelSimulator
import time
import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal
from IPython.display import clear_output


class ACNN(nn.Module):
    """
    Defining an NN architecture to be used for the actor critic approximations
    """

    def __init__(self, in_dim, out_dim):
        """
            Initialize the network and set up the layers.

            Parameters:
                in_dim - input dimensions as an int
                out_dim - output dimensions as an int

            Return:
                None
        """
        super(ACNN, self).__init__()

        self.layer1 = nn.Linear(in_dim, 128)
        self.layer2 = nn.Linear(128, out_dim)

    def forward(self, obs):

        # Convert observation to tensor if it's a numpy array
        if isinstance(obs, tuple):
            obs = np.asarray(obs)
            obs = torch.tensor(obs, dtype=torch.float)
        elif isinstance(obs, np.ndarray):
            obs = torch.tensor(obs, dtype=torch.float)
        elif isinstance(obs, np.int32):
            obs = np.asarray([obs])
            obs = torch.tensor(obs, dtype=torch.float)

        """if obs.ndim == 1:
            obs = torch.reshape(obs, (obs.shape[0], 1))"""
        logits = nn.functional.relu(self.layer1(obs))
        output = self.layer2(logits)

        return output

class PPO:

    def __init__(self, env, alpha=0.1, gamma=0.99):
        self.env = env
        self.obs_dim = 2 # grid world is 10x10
        self.action_dim = 4 # there are four discrete actions that can be taken (up, right, down, left)
        self.actor = ACNN(self.obs_dim, self.action_dim)
        self.critic = ACNN(self.obs_dim, 1)

        self._init_hp(alpha, gamma)
        self.cov_var = torch.full(size=(self.action_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

        self.a_optim = Adam(self.actor.parameters(), lr=self.alpha)
        self.c_optim = Adam(self.critic.parameters(), lr=self.alpha)

        self.logger = {
            'delta_t': time.time_ns(),
            't_so_far': 0,  # timesteps so far
            'i_so_far': 0,  # iterations so far
            'batch_lens': [],  # episodic lengths in batch
            'batch_rews': [],  # episodic returns in batch
            'actor_losses': [],  # losses of actor network in current iteration
            'overall_loss': [],
            'overall_reward': [],
            'batch_accuracy': []
        }

        ACTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
        data = []
        with open('../dataset/minigrid_RLHF_dataset.csv', 'r') as f:
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
        self.test = np.array(data)

    def learn(self, n_episodes):
        t = 0 # t = current time step
        i_so_far = 0
        while t < n_episodes:
            batch_s, batch_a, batch_log_probs, batch_rtgs, batch_lens = self.rollout()
            t += np.sum(batch_lens)
            v, _ = self.evaluate(batch_s, batch_a)

            self.logger['batch_accuracy'].append(self.get_accuracy())

            # Increment the number of iterations
            i_so_far += 1

            # Logging timesteps so far and iterations so far
            self.logger['t_so_far'] = t
            self.logger['i_so_far'] = i_so_far

            adv_k = batch_rtgs - v.detach() # getting the advantages for the time steps

            adv_k = (adv_k - adv_k.mean()) / (adv_k.std() + 1e-10) # normalizing the advantages

            for i in range(self.n_updates_per_iteration):
                v, curr_log_probs = self.evaluate(batch_s, batch_a)

                ratios = torch.exp(curr_log_probs - batch_log_probs)

                s1 = ratios * adv_k
                s2 = torch.clamp(ratios, 1-self.clip, 1 + self.clip)

                actor_loss = (-torch.min(s1, s2)).mean()
                self.a_optim.zero_grad()
                actor_loss.backward()
                self.a_optim.step()

                critic_loss = nn.MSELoss()(v, batch_log_probs)
                self.c_optim.zero_grad()
                critic_loss.backward()
                self.c_optim.step()

                self.logger['actor_losses'].append(actor_loss.detach())


            self._log_summary()

            # Save our model if it's time
            if i_so_far % self.save_freq == 0:
                torch.save(self.actor.state_dict(), './ppo_actor.pth')
                torch.save(self.critic.state_dict(), './ppo_critic.pth')

    def _init_hp(self, alpha, gamma):
        self.steps_per_batch = 10000
        self.max_episode = 500 # maximum timesteps per episode: prevents episode from runnning forever
        self.gamma = gamma
        self.n_updates_per_iteration = 10
        self.clip = 0.2
        self.alpha = alpha
        # Miscellaneous parameters
        self.render = True  # If we should render during rollout
        self.render_every_i = 10  # Only render every n iterations
        self.save_freq = 10  # How often we save in number of iterations
        self.seed = None

    def rollout(self):
        batch_s = []
        batch_a = []
        batch_log_probs = []  # log probs of each action
        batch_r = []
        batch_lens = []  # length of each episode per batch

        batch_t = 0

        while batch_t < self.steps_per_batch:

            episode_r = []
            input = self.env.reset()
            done = False

            for ep_step in range(self.max_episode):
                batch_t += 1
                batch_s.append(input)

                a, log_prob = self.get_action(input)
                state, reward, done, _ = self.env.step(a)

                episode_r.append(reward)
                batch_a.append(a.tolist())
                batch_log_probs.append(log_prob.numpy().tolist())
                if done: break
            batch_lens.append(ep_step + 1)  # plus 1 because timestep starts at 0
            batch_r.append(episode_r)

        batch_s = torch.tensor(batch_s, dtype=torch.float)
        batch_a = torch.tensor(batch_a, dtype=torch.float)
        batch_log_probs = torch.tensor(batch_log_probs, dtype=torch.float)
        batch_rtgs = self.rtgs_comp(batch_r)

        # Log the episodic returns and episodic lengths in this batch.
        self.logger['batch_rews'] = batch_r
        self.logger['batch_lens'] = batch_lens

        return batch_s, batch_a, batch_log_probs, batch_rtgs, batch_lens

    def get_action(self, s):
        mean = self.actor(s) # get mean action from the actor network

        dist = MultivariateNormal(mean, self.cov_mat) # get multivariate distribution to help with exploring
        a = dist.sample()
        log_prob = dist.log_prob(a)
        if a.ndim == 1:
            a = a.argmax()
        else:
            a = a.argmax(dim=1)
        return a.detach().numpy(), log_prob.detach()

    def rtgs_comp(self, batch_r):
        batch_rtgs = []
        for ep_r in reversed(batch_r):
            discount_r = 0

            for r in reversed(ep_r):
                discount_r = r + discount_r * self.gamma
                batch_rtgs.insert(0, discount_r)
        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float)
        return batch_rtgs

    def evaluate(self, batch_obs, batch_acts):
        V = self.critic(batch_obs).squeeze()

        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        a = dist.sample()
        log_probs = dist.log_prob(a)

        return V, log_probs

    def _log_summary(self):
        """
            Print to stdout what we've logged so far in the most recent batch.

            Parameters:
                None

            Return:
                None
        """
        # Calculate logging values. I use a few python shortcuts to calculate each value
        # without explaining since it's not too important to PPO; feel free to look it over,
        # and if you have any questions you can email me (look at bottom of README)
        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        t_so_far = self.logger['t_so_far']
        i_so_far = self.logger['i_so_far']
        avg_ep_lens = np.mean(self.logger['batch_lens'])
        avg_ep_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])
        avg_actor_loss = np.mean([losses.float().mean() for losses in self.logger['actor_losses']])


        self.logger['overall_reward'].append(avg_ep_rews)
        self.logger['overall_loss'].append(avg_actor_loss)

        # Round decimal places for more aesthetic logging messages
        avg_ep_lens = str(round(avg_ep_lens, 2))
        avg_ep_rews = str(round(avg_ep_rews, 2))
        avg_actor_loss = str(round(avg_actor_loss, 5))

        # Print logging statements
        print(flush=True)
        print(f"-------------------- Iteration #{i_so_far} --------------------", flush=True)
        print(f"Average Episodic Length: {avg_ep_lens}", flush=True)
        print(f"Average Episodic Return: {avg_ep_rews}", flush=True)
        print(f"Average Loss: {avg_actor_loss}", flush=True)
        print(f"Average Accuracy: {self.logger['batch_accuracy'][-1]}")
        print(f"Timesteps So Far: {t_so_far}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print(f"------------------------------------------------------", flush=True)
        print(flush=True)

        # Reset batch-specific logging data
        self.logger['batch_lens'] = []
        self.logger['batch_rews'] = []
        self.logger['actor_losses'] = []

    def get_accuracy(self):
        correct = 0
        counter = 0
        np.random.shuffle(self.test)
        for data in self.test[:100000]:

            opt1 = data[0]
            opt2 = data[1]

            s = opt1[:2]

            a, log_prob = self.get_action(s)

            if a == opt1[2]:
                correct += 1
                counter += 1
            elif a == opt2[2]:
                counter += 1
        return correct/counter



def train(env, alpha, gamma, n_steps=200000):

    print("Beginning training procedure")
    model = PPO(env, alpha, gamma)
    model.learn(n_steps)

    plt.plot(range(len(model.logger['overall_loss'])), model.logger['overall_loss'])
    plt.title("Loss of PPO over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Average Epoch loss")
    plt.ylim(top=30)
    plt.ylim(bottom=0)
    plt.savefig(f"../model_plots/ppo_loss_over_epochs_lr_{model.alpha}_gamma_{model.gamma}.png")
    plt.close()

    plt.plot(range(len(model.logger['overall_reward'])), model.logger['overall_reward'])
    plt.title("Reward of PPO over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Average Epoch Reward")
    plt.savefig(f"../model_plots/ppo_reward_over_epochs_lr_{model.alpha}_gamma_{model.gamma}.png")
    plt.close()

    plt.plot(range(len(model.logger['batch_accuracy'])), model.logger['batch_accuracy'])
    plt.title("Accuracy of PPO over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.savefig(f"../model_plots/ppo_accuracy_over_epochs_lr_{model.alpha}_gamma_{model.gamma}.png")
    plt.close()

    return model.logger['overall_loss'], model.logger['overall_reward'], model.logger['batch_accuracy']
def test(env, actor_model):

    print(f"Testing {actor_model}", flush=True)

    # If the actor model is not specified, then exit
    if actor_model == '':
        raise Exception(f"Didn't specify model file. Exiting.", flush=True)

    # Extract out dimensions of observation and action spaces
    obs_dim = 2
    act_dim = 4

    # Build our policy the same way we build our actor model in PPO
    policy = ACNN(obs_dim, act_dim)

    # Load in the actor model saved by the PPO algorithm
    policy.load_state_dict(torch.load(actor_model))

    # Evaluate our policy with a separate module, eval_policy, to demonstrate
    # that once we are done training the model/policy with ppo.py, we no longer need
    # ppo.py since it only contains the training algorithm. The model/policy itself exists
    # independently as a binary file that can be loaded in with torch.
    eval_policy(policy=policy, env=env, render=True)

def train_set():
    env = RewardModelSimulator()

    losses = {}
    rewards = {}
    for alpha in [0.005, 0.01, 0.02, 0.05]:
        for gamma in [0.925]:
            l, r = train(env, alpha, gamma, 200000)

            losses[f"{alpha}:{gamma}"] = l
            rewards[f"{alpha}:{gamma}"] = r


    for loss in losses:
        plt.plot(losses[loss], label=loss)
    plt.title("Loss of PPO over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Average Epoch loss")
    plt.legend()
    plt.savefig(f"../model_plots/combined_PPO_loss_over_epochs.png")
    plt.close()

    for reward in rewards:
        plt.plot(rewards[reward], label=reward)
    plt.title("Reward of PPO over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Average Epoch Reward")
    plt.legend()
    plt.savefig(f"../model_plots/combined_PPO_reward_over_epochs.png")
    plt.close()
    #test(env, actor_model='ppo_actor.pth')

env = RewardModelSimulator()

train(env, 0.02, 0.925, 200000)
