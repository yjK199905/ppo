import sys
import argparse

def get_args():
    sys.arv = ['']
    parser = argparse.ArgumentParser()

    parser.add_argument('--env', dest='env', type=str, default='')
    parser.add_argument('--mode', dest='mode', type=str, default='train')
    parser.add_argument('--actor', dest='actor', type=str, default='')
    parser.add_argument('--critic', dest='critic', type=str, default='')
    parser.add_argument('--num_epi', type=int, default=100)

    args = parser.parse_args()
    return args