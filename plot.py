import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

from arg import get_args
args = get_args()

def plot_train(trn_data, save_path=f"metrics/{args.env}_trn.png"):
    episodes = list(map(int, trn_data['episode']))
    ep_lens = list(map(float, trn_data['ep_len']))
    ep_rets = list(map(float, trn_data['ep_ret']))
    actor_losses = list(map(float, trn_data['actor_loss']))
    critic_losses = list(map(float, trn_data['critic_loss']))

    plt.figure(figsize=(14, 10))

    plt.subplot(2, 2, 1)
    plt.plot(episodes, ep_lens, label="Episode Length", color="black", linewidth=2, linestyle='-', marker='o', markersize=5)
    plt.ylabel("Length", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper left', fontsize=12)

    plt.subplot(2, 2, 2)
    plt.plot(episodes, ep_rets, label="Episode Return", color="orange", linewidth=2, linestyle='-', marker='x', markersize=5)
    plt.ylabel("Return", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper left', fontsize=12)

    plt.subplot(2, 2, 3)
    plt.plot(episodes, actor_losses, label="Actor Loss", color="red", linewidth=2, linestyle='-', marker='s', markersize=5)
    plt.xlabel("Episode", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right', fontsize=12)

    plt.subplot(2, 2, 4)
    plt.plot(episodes, critic_losses, label="Critic Loss", color="blue", linewidth=2, linestyle='-', marker='^', markersize=5)
    plt.xlabel("Episode", fontsize=14)
    plt.ylabel("Loss", fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(loc='upper right', fontsize=12)

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Training metrics saved to {save_path}")

def plot_test(test_data, save_path=f"metrics/{args.env}_test.png"):
    ep_rets = [data['ep_ret'] for data in test_data]  
    actor_losses = [data['actor_loss'].item() for data in test_data]  
    critic_losses = [data['critic_loss'].item() for data in test_data]

    plt.figure(figsize=(18, 6))

    plt.subplot(1, 3, 1)
    sns.kdeplot(ep_rets, color="orange", fill=True, alpha=0.6)
    plt.title("Distribution of Return", fontsize=18)
    plt.ylabel("")
    
    plt.subplot(1, 3, 2)
    sns.kdeplot(actor_losses, color="red", fill=True, alpha=0.6)
    plt.title("Distribution of Actor Loss", fontsize=18)
    plt.ylabel("")

    plt.subplot(1, 3, 3)
    sns.kdeplot(critic_losses, color="blue", fill=True, alpha=0.6)
    plt.title("Distribution of Critic Loss", fontsize=18)
    plt.ylabel("")

    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Test metrics saved to {save_path}")