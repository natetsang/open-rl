import matplotlib.pyplot as plt
import numpy as np
from typing import List


def plot_training_results(mean_rewards_history: List,
                          max_rew_history: List,
                          min_rew_history: List,
                          mean_loss_history: List,
                          max_loss_history: List,
                          min_loss_history: List,
                          steps_history: List,
                          wallclock_history: List,
                          save_dir: str = None):

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    fig.suptitle("Results", fontsize=16)

    # Epochs vs rewards
    num_epochs = len(mean_rewards_history)
    ax1.plot(np.arange(num_epochs), mean_rewards_history, color='green', label="mean epoch reward")
    ax1.plot(np.arange(num_epochs), min_rew_history, color='grey', alpha=0.5, linestyle='-.')
    ax1.plot(np.arange(num_epochs), max_rew_history, color='grey', alpha=0.5, linestyle='-.')

    ax1.set_xlabel("Number of epochs")
    ax1.set_ylabel("Reward")
    ax1.legend()
    ax1.grid()

    # Steps vs rewards
    cumulative_steps = list(np.cumsum(steps_history))
    ax2.plot(cumulative_steps, mean_rewards_history, color='green', label="mean epoch reward")
    ax2.set_xlabel("Number of steps")
    ax2.set_ylabel("Reward")
    ax2.legend()
    ax2.grid()

    # Epochs vs loss
    ax3.plot(np.arange(num_epochs), mean_loss_history, color='green', label="mean epoch loss")
    ax3.plot(np.arange(num_epochs), min_loss_history, color='grey', alpha=0.5, ls='-.')
    ax3.plot(np.arange(num_epochs), max_loss_history, color='grey', alpha=0.5, linestyle='-.')
    ax3.set_xlabel("Number of epochs")
    ax3.set_ylabel("Loss")
    ax3.legend()
    ax3.grid()

    # Wallclock vs rewards
    ax4.plot(wallclock_history, mean_rewards_history, color='green', label="epoch reward")
    ax4.set_xlabel("Wallclock time (seconds)")
    ax4.set_ylabel("Reward")
    ax4.legend()
    ax4.grid()

    fig.tight_layout()
    fig.subplots_adjust(top=0.94)
    plt.show()

    if save_dir:
        plt.savefig(save_dir)
