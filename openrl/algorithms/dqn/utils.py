import matplotlib.pyplot as plt
import numpy as np
from typing import List
import random
from collections import deque


class PrioritizedReplayBuffer:
    def __init__(self, maxlen):
        self.buffer = deque(maxlen=maxlen)
        self.priorities = deque(maxlen=maxlen)

    def add(self, experience):
        self.buffer.append(experience)
        self.priorities.append(max(self.priorities, default=1))

    def get_probabilities(self, priority_scale):
        scaled_priorities = np.array(self.priorities) ** priority_scale
        sample_probabilities = scaled_priorities / sum(scaled_priorities)
        return sample_probabilities

    def get_importance(self, probabilities):
        importance = 1 / len(self.buffer) * 1 / probabilities
        importance_normalized = importance / max(importance)
        return importance_normalized

    def sample(self, batch_size, priority_scale=1.0):
        sample_size = min(len(self.buffer), batch_size)
        sample_probs = self.get_probabilities(priority_scale)
        sample_indices = random.choices(range(len(self.buffer)), k=sample_size, weights=sample_probs)
        samples = np.array(self.buffer)[sample_indices]
        importance = self.get_importance(sample_probs[sample_indices])
        return map(list, zip(*samples)), importance, sample_indices

    def set_priorities(self, indices, errors, offset=0.1):
        for i, e in zip(indices, errors):
            self.priorities[i] = abs(e) + offset


def plot_training_results(rewards_history: List,
                          running_rewards_history: List,
                          steps_history: List,
                          wallclock_history: List,
                          loss_history: List = None,
                          save_dir: str = None):

    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) = plt.subplots(nrows=3, ncols=3, figsize=(16, 12))
    fig.suptitle("Training Results", fontsize=16)

    # EPISODES
    num_episodes = len(rewards_history)

    # Episodes vs rewards
    ax1.plot(np.arange(num_episodes), rewards_history, color='grey', alpha=0.3, linestyle='-', label="episode reward")
    ax1.plot(np.arange(num_episodes), running_rewards_history, color='green', label="moving average reward")
    ax1.set_xlabel("Number of episodes")
    ax1.set_ylabel("Episode reward")
    ax1.legend()
    ax1.grid()

    # Episodes vs loss
    if loss_history:
        ax4.plot(np.arange(num_episodes), loss_history, color='green')
        ax4.set_xlabel("Number of episodes")
        ax4.set_ylabel("Episode loss")
        ax4.grid()
    else:
        ax4.set_axis_off()

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
    ax2.set_ylabel("Episode reward")
    ax2.legend()
    ax2.grid()

    # Steps vs loss
    if loss_history:
        ax5.plot(cumulative_steps, loss_history, color='green')
        ax5.set_xlabel("Number of steps")
        ax5.set_ylabel("Episode loss")
        ax5.grid()
    else:
        ax5.set_axis_off()

    # TIME

    # Wallclock vs rewards
    ax3.plot(wallclock_history, rewards_history, color='grey', alpha=0.3, linestyle='-', label="episode reward")
    ax3.plot(wallclock_history, running_rewards_history, color='green', label="moving average reward")
    ax3.set_xlabel("Wallclock time (seconds)")
    ax3.set_ylabel("Episode reward")
    ax3.legend()
    ax3.grid()

    # Wallclock vs loss
    if loss_history:
        ax6.plot(wallclock_history, loss_history, color='green')
        ax6.set_xlabel("Wallclock time (seconds)")
        ax6.set_ylabel("Episode loss")
        ax6.grid()
    else:
        ax6.set_axis_off()

    # Turn off unused plots
    ax8.set_axis_off()
    ax9.set_axis_off()

    # Save
    if save_dir:
        plt.savefig(save_dir)

    # Show plots
    plt.tight_layout()
    plt.show()
