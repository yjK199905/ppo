import sys
import gymnasium as gym

from arg import get_args
from main import train
from main import test

def main(args):
    params = {
        'batch_steps': 1024,
        'epi_steps': 1000,
        'gamma': 0.999,
        'num_update': 4,
        'lr': 2.5e-4,
        'clip': 0.2
    }

    env = gym.make('LunarLanderContinuous-v2', render_mode="rgb_array")

    if args.mode == 'train':
        train(env=env, params=params, actor=args.actor, critic=args.critic)
    elif args.mode == 'test':
        test(env=env, actor=args.actor, critic=args.critic, num_epi=args.num_epi)
    else:
        print("You specified wrong mode name.")
        sys.exit(0)

if __name__ == '__main__':
    args = get_args()
    main(args)