import sys
import torch
import imageio
import gymnasium as gym

from arg import get_args
from ppo import PPO
from net import Network
from eval import eval_pi
from plot import plot_train, plot_test

def train(env, params, actor, critic):
    model = PPO(network=Network, env=env, **params)
    
    if actor != '' and critic != '':
        print(f"Loading {actor} and {critic}...", flush=True)
        model.actor.load_state_dict(torch.load(actor))
        model.critic.load_state_dict(torch.load(critic))
        print(f"Successfully loaded.", flush=True)
    elif actor != '' or critic != '':
        print(f"Specify both actor/critic or none at all.")
        sys.exit(0)
    else:
        print(f"Training from scratch.", flush=True)

    args = get_args()
    model.learn(total_steps=args.num_epi)
    
    plot_train(model.trn_data)

def test(env, actor, critic, num_epi=100):
    print(f"Testing {actor}...", flush=True)

    if actor == '' or critic == '':
        print(f"Specify actor and critic network.", flush=True)
        sys.exit(0)

    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.shape[0]

    pi = Network(obs_dim, act_dim)
    pi.load_state_dict(torch.load(actor))

    critic_model = Network(obs_dim, 1)
    critic_model.load_state_dict(torch.load(critic))

    test_data, best_gif = list(eval_pi(pi=pi, env=env, critic=critic_model, num_epi=num_epi))
    plot_test(test_data)

    args = get_args()
    if best_gif:
        save_gif(best_gif, f"gif/{args.env}_best.gif")
        print("Best episode GIF is saved.")

def save_gif(gifs, name, f_dur=0.03, optimize=True, loop=0):
    ex_gifs = []
    rep = 3
    for gif in gifs:
        ex_gifs.extend([gif] * rep)

    with imageio.get_writer(name, mode="I", duration=f_dur, loop=loop) as writer:
        for _, gif in enumerate(gifs):
            if gif.shape[-1] != 3:
                gif = gif[..., :3]
            writer.append_data(gif)
    print(f"Saved GIF: {name}")