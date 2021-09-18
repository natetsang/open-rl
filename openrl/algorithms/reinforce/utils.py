import matplotlib.pyplot as plt
import numpy as np
from typing import List


def plot_training_results(rewards_history: List,
                          running_rewards_history: List,
                          steps_history: List,
                          loss_history: List,
                          wallclock_history: List,
                          save_dir: str = None):

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(nrows=3, ncols=3, figsize=(16, 12))
    fig.suptitle("Training Results", fontsize=16)

    # EPISODES
    num_episodes = len(rewards_history)

    # Episodes vs rewards
    ax1.plot(np.arange(num_episodes), rewards_history, color='grey', alpha=0.3, linestyle='-', label="episode reward")
    ax1.plot(np.arange(num_episodes), running_rewards_history, color='green', label="moving average reward")
    ax1.set_xlabel("Number of episodes")
    ax1.set_ylabel("Reward")
    ax1.legend()
    ax1.grid()

    # Episodes vs loss
    ax4.plot(np.arange(num_episodes), loss_history, color='green')
    ax4.set_xlabel("Number of episodes")
    ax4.set_ylabel("Loss")
    ax4.grid()

    # Epochs vs steps per epoch
    ax7.plot(np.arange(num_episodes), steps_history, color='green')
    ax7.set_xlabel("Number of episodes")
    ax7.set_ylabel("Episode duration (steps)")
    ax7.grid()

    # STEPS
    cumulative_steps = list(np.cumsum(steps_history))

    # Steps vs rewards
    ax2.plot(cumulative_steps, rewards_history, color='grey', alpha=0.3, linestyle='-', label="episode reward")
    ax2.plot(cumulative_steps, running_rewards_history, color='green', label="moving average reward")
    ax2.set_xlabel("Number of steps")
    ax2.set_ylabel("Reward")
    ax2.legend()
    ax2.grid()

    # Steps vs loss
    ax5.plot(cumulative_steps, loss_history, color='green')
    ax5.set_xlabel("Number of steps")
    ax5.set_ylabel("Loss")
    ax5.grid()

    # TIME

    # Wallclock vs rewards
    ax3.plot(wallclock_history, rewards_history, color='grey', alpha=0.3, linestyle='-', label="episode reward")
    ax3.plot(wallclock_history, running_rewards_history, color='green', label="moving average reward")
    ax3.set_xlabel("Wallclock time (seconds)")
    ax3.set_ylabel("Reward")
    ax3.legend()
    ax3.grid()

    # Wallclock vs loss
    ax6.plot(wallclock_history, loss_history, color='green')
    ax6.set_xlabel("Wallclock time (seconds)")
    ax6.set_ylabel("Loss")
    ax6.grid()

    # Turn off unused plots
    ax8.set_axis_off()
    ax9.set_axis_off()

    # Save
    if save_dir:
        plt.savefig(save_dir)

    # Show plots
    plt.tight_layout()
    plt.show()
