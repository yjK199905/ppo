import time
import numpy as np
import gymnasium as gym

import torch
import torch.nn as nn

from arg import get_args
from torch.optim import Adam
from torch.distributions import MultivariateNormal

class PPO:
    def __init__(self, network, env, **params):
        assert(type(env.observation_space) == gym.spaces.Box)
        assert(type(env.action_space) == gym.spaces.Box)

        self._init_params(params)
        
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]

        self.actor = network(self.obs_dim, self.act_dim)
        self.critic = network(self.obs_dim, 1)
        self.actor_optim = Adam(self.actor.parameters())
        self.critic_optim = Adam(self.critic.parameters())

        self.cov_var = torch.full(size=(self.act_dim,), fill_value=0.5)
        self.cov_mat = torch.diag(self.cov_var)
        self.logger = {
            'delta_t': time.time_ns(),
            'step': 0,
            'iter': 0,
            'batch_lens': [],
            'batch_Rs': [],
            'actor_Ls': [],
            'critic_Ls': []
        }
        self.trn_data = {
            'episode': [],
            'ep_len': [],
            'ep_ret': [],
            'actor_loss': [],
            'critic_loss': []
        }

    def learn(self, total_steps):
        print(f"Total timesteps: {total_steps}.")
        print(f"Max step per episode: {self.epi_steps}, Steps per batch: {self.batch_steps}.")

        step = 0
        iter = 0
        while step < total_steps:
            batch_obs, batch_acts, batch_probs, batch_rtgs, batch_lens = self.rollout()

            step += np.sum(batch_lens)
            iter += 1

            self.logger['step'] = step
            self.logger['iter'] = iter

            V, _ = self.evaluate(batch_obs, batch_acts)
            A_k = batch_rtgs - V.detach()
            A_k = (A_k - A_k.mean()) / (A_k.std() + 1e-10)

            for _ in range(self.num_update):
                V, cur_probs = self.evaluate(batch_obs, batch_acts)

                ratios = torch.exp(cur_probs - batch_probs)

                surr1 = ratios * A_k
                surr2 = torch.clamp(ratios, 1 - self.clip, 1 + self.clip) * A_k

                actor_loss = (-torch.min(surr1, surr2)).mean()
                critic_loss = nn.MSELoss()(V, batch_rtgs)

                self.actor_optim.zero_grad()
                actor_loss.backward(retain_graph=True)
                self.actor_optim.step()

                self.critic_optim.zero_grad()
                critic_loss.backward(retain_graph=True)
                self.critic_optim.step()

                self.logger['actor_Ls'].append(actor_loss.detach())
                self.logger['critic_Ls'].append(critic_loss.detach())

            avg_actor_loss = np.mean([loss.float().mean() for loss in self.logger['actor_Ls']])
            avg_critic_loss = np.mean([loss.float().mean() for loss in self.logger['critic_Ls']])
            
            self._log_summary(avg_actor_loss, avg_critic_loss)

            args = get_args()
            if iter % self.save_freq == 0:
                torch.save(self.actor.state_dict(), f'pths/{args.env}_actor.pth')
                torch.save(self.critic.state_dict(), f'pths/{args.env}_critic.pth')

        print("Training complete.")

    def rollout(self):
        batch_obs, batch_acts, batch_probs, batch_rews, batch_rtgs, batch_lens = [], [], [], [], [], []
        ep_rews = []

        t = 0
        while t < self.batch_steps:
            ep_rews = []

            obs, _ = self.env.reset()
            done = False

            for ep_t in range(self.epi_steps):
                t += 1

                batch_obs.append(obs)

                act, prob = self.get_action(obs)
                obs, rew, term, trunc, _ = self.env.step(act)
                done = term or trunc

                ep_rews.append(rew)
                batch_acts.append(act)
                batch_probs.append(prob)

                if done:
                    break

            batch_lens.append(ep_t + 1)
            batch_rews.append(ep_rews)

        batch_obs = np.array(batch_obs)
        batch_obs = torch.tensor(batch_obs, dtype=torch.float32)
        batch_acts = np.array(batch_acts)
        batch_acts = torch.tensor(batch_acts, dtype=torch.float32)
        batch_probs = torch.tensor(batch_probs, dtype=torch.float32)
        batch_rtgs = self.compute_rtgs(batch_rews)

        self.logger['batch_rews'] = batch_rews
        self.logger['batch_lens'] = batch_lens

        return batch_obs, batch_acts, batch_probs, batch_rtgs, batch_lens
    
    def compute_rtgs(self, batch_rews):
        batch_rtgs = []
        for ep_rews in reversed(batch_rews):
            dc_rew = 0

            for rew in reversed(ep_rews):
                dc_rew = rew + dc_rew * self.gamma
                batch_rtgs.insert(0, dc_rew)

        batch_rtgs = torch.tensor(batch_rtgs, dtype=torch.float32)
        return batch_rtgs
    
    def get_action(self, obs):
        mean = self.actor(obs)
        dist = MultivariateNormal(mean, self.cov_mat)

        act = dist.sample()
        prob = dist.log_prob(act)
        return act.detach().numpy(), prob.detach()
    
    def evaluate(self, batch_obs, batch_acts):
        V = self.critic(batch_obs).squeeze()

        mean = self.actor(batch_obs)
        dist = MultivariateNormal(mean, self.cov_mat)
        prob = dist.log_prob(batch_acts)

        return V, prob
    
    def _init_params(self, params):
        self.batch_steps = 4800
        self.epi_steps = 1600
        self.num_update = 5
        self.lr = 0.005
        self.gamma = 0.95
        self.clip = 0.2

        self.save_freq = 10
        self.seed = None

        for param, val in params.items():
            setattr(self, param, val)

        if self.seed != None:
            assert(type(self.seed) == int)

            torch.manual_seed(self.seed)
            print(f"Successfully set seed to {self.seed}")

    def _log_summary(self, avg_actor_loss, avg_critic_loss):
        delta_t = self.logger['delta_t']
        self.logger['delta_t'] = time.time_ns()
        delta_t = (self.logger['delta_t'] - delta_t) / 1e9
        delta_t = str(round(delta_t, 2))

        step = self.logger['step']
        iter = self.logger['iter']
        avg_lens = np.mean(self.logger['batch_lens'])
        avg_rews = np.mean([np.sum(ep_rews) for ep_rews in self.logger['batch_rews']])

        avg_lens = str(round(avg_lens, 2))
        avg_rews = str(round(avg_rews, 2))

        self.trn_data['episode'].append(self.logger['iter'])
        self.trn_data['ep_len'].append(avg_lens)
        self.trn_data['ep_ret'].append(avg_rews)
        self.trn_data['actor_loss'].append(avg_actor_loss)
        self.trn_data['critic_loss'].append(avg_critic_loss)

        print(flush=True)
        print(f"--------------------Iteration #{iter}-------------------------")
        print(f"Average Episode Length: {avg_lens}", flush=True)
        print(f"Average Episode Return: {avg_rews}", flush=True)
        print(f"Average Actor Loss: {avg_actor_loss:.6f}", flush=True)
        print(f"Average Critic Loss: {avg_critic_loss:.6f}", flush=True)
        print(f"Timesteps So Far: {step}", flush=True)
        print(f"Iteration took: {delta_t} secs", flush=True)
        print(f"-----------------------------------------------------------", flush=True)
        print(flush=True)

        self.logger['batch_lens'] = []
        self.logger['batch_rews'] = []
        self.logger['actor_losses'] = []