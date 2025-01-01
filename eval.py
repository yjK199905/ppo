import numpy as np

import torch
import torch.nn as nn

from torch.distributions import MultivariateNormal

def _log_summary(ep_len, ep_ret, ep_num, actor_loss, critic_loss):
    ep_len = str(round(ep_len, 2))
    ep_ret = str(round(ep_ret, 2))

    print(flush=True)
    print(f"-------------------- Episode #{ep_num+1} --------------------", flush=True)
    print(f"Episode Length: {ep_len}", flush=True)
    print(f"Episode Return: {ep_ret}", flush=True)
    print(f"Actor Loss: {actor_loss:.5f}", flush=True)
    print(f"Critic Loss: {critic_loss:.5f}", flush=True)
    print(f"------------------------------------------------------", flush=True)
    print(flush=True)

def rollout(pi, env, critic=None):
    while True:
        obs, _ = env.reset()
        done = False

        t = 0
        ep_len = 0
        ep_ret = 0
        gifs = []

        actor_loss = None
        critic_loss = None

        while not done:
            t += 1

            if t % 10 == 0:
                frame = env.render()
                gifs.append(frame)

            act_mean = pi(torch.tensor(obs, dtype=torch.float32)).detach().numpy()
            noise = np.random.normal(scale=0.1, size=act_mean.shape)
            act = act_mean + noise
            
            obs, rew, term, trunc, _ = env.step(act)
            done = term or trunc

            ep_ret += rew

        if critic:
            obs_tensor = torch.tensor(obs, dtype=torch.float32).unsqueeze(0)
            act_tensor = torch.tensor(act, dtype=torch.float32).unsqueeze(0)
            
            critic_value = critic(obs_tensor).squeeze()
            target_value = torch.tensor([ep_ret], dtype=torch.float32).squeeze()
            critic_loss = nn.MSELoss()(critic_value, target_value)

            dist = MultivariateNormal(pi(obs_tensor), torch.eye(pi(obs_tensor).size(-1)))
            log_prob = dist.log_prob(act_tensor)
            actor_loss = -log_prob.mean()
        
        ep_len = t
        yield ep_len, ep_ret, actor_loss, critic_loss, gifs

def eval_pi(pi, env, critic=None, num_epi=100):
    best_ret = -float('inf')
    best_gif = None
    test_data = []

    for ep_num, (ep_len, ep_ret, actor_loss, critic_loss, gifs) in enumerate(rollout(pi, env, critic)):
        if actor_loss is not None:
            actor_loss = actor_loss.detach()
        if critic_loss is not None:
            critic_loss = critic_loss.detach()
        _log_summary(ep_len=ep_len, ep_ret=ep_ret, ep_num=ep_num, actor_loss=actor_loss, critic_loss=critic_loss)
        test_data.append({
            'ep_len': ep_len,
            'ep_ret': ep_ret,
            'actor_loss': actor_loss,
            'critic_loss': critic_loss
        })
        
        if ep_ret > best_ret:
            best_ret = ep_ret
            best_gif = gifs
        
        if ep_num + 1 >= num_epi:
            break

    return test_data, best_gif