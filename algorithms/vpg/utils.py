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

    fig.tight_layout()
    fig.subplots_adjust(top=0.94)
    plt.show()

    if save_dir:
        plt.savefig(save_dir)




# TODO >> Add steps and time as single graph but multiple axes
#   Figure out how to make the x-axes align correctly
# def plot_training_results_new(rewards_history: List,
#                           running_rewards_history: List,
#                           steps_history: List,
#                           wallclock_history: List,
#                           save_dir: str = None):
#
#     cumulative_steps = list(np.cumsum(steps_history))
#
#     fig, (ax1) = plt.subplots(1, 1, figsize=(12, 8))
#     fig.suptitle("Rewards history")
#
#     num_epochs = len(rewards_history)
#     newax = ax1.twiny()
#     # newax2 = ax1.twiny()
#
#     # Make some room at the bottom
#     fig.subplots_adjust(bottom=0.20)
#
#     newax.set_frame_on(True)
#     newax.patch.set_visible(False)
#     newax.xaxis.set_ticks_position('bottom')
#     newax.xaxis.set_label_position('bottom')
#     newax.spines['bottom'].set_position(('outward', 40))
#     newax.set_xlim([cumulative_steps[0], cumulative_steps[-1]])
#     xmin, xmax = newax.get_xlim()
#     newax.set_xticks(np.round(np.linspace(xmin, xmax, 5), 2))
#
#     # newax2.set_frame_on(True)
#     # newax2.patch.set_visible(False)
#     # newax2.xaxis.set_ticks_position('bottom')
#     # newax2.xaxis.set_label_position('bottom')
#     # newax2.spines['bottom'].set_position(('outward', 80))
#     # newax2.set_xlim([wallclock_history[0], wallclock_history[-1]])
#
#
#     # ax1.plot(np.arange(num_epochs), rewards_history, color='blue', alpha=0.3, linestyle='-', label="episode reward")
#     ax1.plot(np.arange(num_epochs), running_rewards_history, color='green', label="running average")
#     ax1.set_ylabel("Reward")
#     ax1.set_xlabel("Epoch #")
#     ax1.set_xlim([0, num_epochs])
#     xmin, xmax = ax1.get_xlim()
#     ax1.set_xticks(np.round(np.linspace(xmin, xmax, 5), 2))
#     newax.plot(cumulative_steps, running_rewards_history, color='red')
#     newax.set_xlabel('Green Thing')
#
#     # newax2.plot(wallclock_history, running_rewards_history, color='black')
#     # newax2.set_xlabel('Blue Thing')
#     assert len(np.arange(num_epochs)) == len(cumulative_steps) == len(wallclock_history)
#
#     for epoch, step in zip(np.arange(num_epochs), cumulative_steps):
#         print(f"EPOCH: {epoch} --- STEP: {step}")
#
#     plt.show()