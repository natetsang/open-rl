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
    ## For a N batch-size, this is steps per epoch, which has N trajectories...It's not steps per trajectory unless N=1
    ax4.plot(np.arange(num_epochs), steps_history)
    ax4.set_xlabel("Number of steps")
    ax4.set_ylabel("Episode duration (steps)")
    ax4.grid()

    fig.tight_layout()
    fig.subplots_adjust(top=0.94)
    plt.show()

    if save_dir:
        plt.savefig(save_dir)

# def plot_training_results(rewards_history: List,
#                           running_rewards_history: List,
#                           num_steps: int = None,
#                           save_dir: str = None):
#
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(6, 8))
#     fig.suptitle("Rewards history")
#
#     num_epochs = len(rewards_history)
#     ax1.plot(np.arange(num_epochs), rewards_history, color='grey', alpha=0.3, linestyle='-', label="episode reward")
#     ax1.plot(np.arange(num_epochs), running_rewards_history, color='green', label="running average")
#     ax1.set_xlabel("Epoch #")
#     ax1.set_ylabel("Reward")
#     ax1.legend()
#
#     # TODO >> Fix this to be for steps
#     if num_steps:
#         ax2.plot(np.arange(num_steps), rewards_history, color='grey', alpha=0.3, linestyle='-', label="step reward")
#         ax2.plot(np.arange(num_steps), running_rewards_history, color='green', label="running average")
#         ax2.set_xlabel("Step #")
#         ax2.set_ylabel("Reward")
#         ax2.legend()
#
#     if save_dir:
#         plt.savefig(save_dir)
#
#     plt.show()
