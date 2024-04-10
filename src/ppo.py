"""
This file contains the class for the PPO algorithm

Inspired by tutorial given by Eric Yang Yu: https://medium.com/analytics-vidhya/coding-ppo-from-scratch-with-pytorch-part-1-4-613dfc1b14c8
"""

import gym
from RLHFenvironment import RLHFEnv

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.distributions import MultivariateNormal


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

    def __init__(self, env):
        self.env = env
        self.obs_dim = 2 # grid world is 10x10
        self.action_dim = 4 # there are four discrete actions that can be taken (up, right, down, left)
        self.actor = ACNN(self.obs_dim, self.action_dim)
        self.critic = ACNN(self.obs_dim, 1)

        self._init_hp()
        self.cov_var = torch.full(size=(self.action_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)

        self.a_optim = Adam(self.actor.parameters(), lr=self.alpha)
        self.c_optim = Adam(self.critic.parameters(), lr=self.alpha)
    def learn(self, n_episodes):
        t = 0 # t = current time step
        while t < n_episodes:
            batch_s, batch_a, batch_log_probs, batch_rtgs, batch_lens = self.rollout()
            t += np.sum(batch_lens)
            v, _ = self.evaluate(batch_s, batch_a)

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


    def _init_hp(self):
        self.steps_per_batch = 4800
        self.max_episode = 1600 # maximum timesteps per episode: prevents episode from runnning forever
        self.gamma = 0.95
        self.n_updates_per_iteration = 5
        self.clip = 0.2
        self.alpha = 0.005

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

env = RLHFEnv()
model = PPO(env)
model.learn(1000)