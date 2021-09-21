import matplotlib.pyplot as plt
import numpy as np
from typing import List


def plot_training_results(rewards_history: List,
                          running_rewards_history: List,
                          steps_history: List,
                          wallclock_history: List,
                          save_dir: str = None):

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    fig.suptitle("Results", fontsize=16)

    # Epochs vs rewards
    num_epochs = len(rewards_history)
    ax1.plot(np.arange(num_epochs), rewards_history, color='grey', alpha=0.3, linestyle='-', label="epoch reward")
    ax1.plot(np.arange(num_epochs), running_rewards_history, color='green', label="moving average reward")
    ax1.set_xlabel("Number of epochs")
    ax1.set_ylabel("Reward")
    ax1.legend()
    ax1.grid()

    # Steps vs rewards
    cumulative_steps = list(np.cumsum(steps_history))

    ax2.plot(cumulative_steps, rewards_history, color='grey', alpha=0.3, linestyle='-', label="epoch reward")
    ax2.plot(cumulative_steps, running_rewards_history, color='green', label="moving average reward")
    ax2.set_xlabel("Number of steps")
    ax2.set_ylabel("Reward")
    ax2.legend()
    ax2.grid()

    # Wallclock vs rewards
    ax3.plot(wallclock_history, rewards_history, color='grey', alpha=0.3, linestyle='-', label="epoch reward")
    ax3.plot(wallclock_history, running_rewards_history, color='green', label="moving average reward")
    ax3.set_xlabel("Wallclock time (seconds)")
    ax3.set_ylabel("Reward")
    ax3.legend()
    ax3.grid()

    # Epochs vs steps per epoch
    ax4.plot(np.arange(num_epochs), steps_history)
    ax4.set_xlabel("Number of steps")
    ax4.set_ylabel("Episode duration (steps)")
    ax4.grid()

    # Save
    if save_dir:
        plt.savefig(save_dir)

    # Show plots
    plt.tight_layout()
    plt.show()
