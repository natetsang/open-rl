import numpy as np
from typing import Tuple, Union, List
import matplotlib.pyplot as plt


class PPOBuffer:
    def __init__(self, state_size, action_size, capacity, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.capacity = capacity
        self.batch_size = batch_size
        self.size = 0
        self.idx = 0
        self._initialize_empty_buffers()

    def _initialize_empty_buffers(self) -> None:
        self.buffer_state = np.zeros(shape=(self.capacity, self.state_size))
        self.buffer_action = np.zeros(shape=(self.capacity, self.action_size))
        self.buffer_reward = np.zeros(shape=(self.capacity, 1))
        self.buffer_mask = np.zeros(shape=(self.capacity, 1))

        self.buffer_value = np.zeros(shape=(self.capacity, 1))
        self.buffer_logp = np.zeros(shape=(self.capacity, 1))
        self.buffer_adv = np.zeros(shape=(self.capacity, 1))
        self.buffer_return = np.zeros(shape=(self.capacity, 1))

    def store_transition(self, transition: Tuple) -> None:
        state, action, reward, mask, value, logp, adv, ret = transition
        # This will make sure to overwrite the oldest transition if full
        current_index = self.idx % self.capacity
        self.buffer_state[current_index] = state
        self.buffer_action[current_index] = action
        self.buffer_reward[current_index] = reward
        self.buffer_mask[current_index] = mask
        self.buffer_value[current_index] = value
        self.buffer_logp[current_index] = logp
        self.buffer_adv[current_index] = adv
        self.buffer_return[current_index] = ret

        # Increment counters
        if self.size < self.capacity:
            self.size += 1
        self.idx += 1

    def all(self) -> Union[None, Tuple]:
        # We can't sample if we don't have enough transitions
        if self.size < self.capacity:
            return

        return (
            self.buffer_state, self.buffer_action, self.buffer_reward,
            self.buffer_mask, self.buffer_value, self.buffer_logp,
            self.buffer_adv, self.buffer_return
                )

    def sample(self) -> Union[None, Tuple]:
        """
        Returns a batch_size of random transitions from the buffer. If there are less
        than batch_size transitions in the buffer, this returns None. Note that sampling
        does not remove the transitions from the buffer so they could be sampled again.
        """

        # We can't sample if we don't have enough transitions
        if self.size < self.capacity:
            return

        idxs = np.random.choice(self.size, self.batch_size)

        batch_state = self.buffer_state[idxs]
        batch_action = self.buffer_action[idxs]
        batch_reward = self.buffer_reward[idxs]
        batch_mask = self.buffer_mask[idxs]
        batch_value = self.buffer_value[idxs]
        batch_logp = self.buffer_logp[idxs]
        batch_adv = self.buffer_adv[idxs]
        batch_return = self.buffer_return[idxs]

        return (batch_state, batch_action, batch_reward, batch_mask,
                batch_value, batch_logp, batch_adv, batch_return)

    def flush(self) -> None:
        self.size = 0
        self.idx = 0
        self._initialize_empty_buffers()

    def __len__(self) -> int:
        return self.size


def plot_training_results(rewards_history: List,
                          running_rewards_history: List,
                          steps_history: List,
                          wallclock_history: List,
                          test_freq: int = 1,
                          save_dir: str = None):

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    fig.suptitle("Results", fontsize=16)

    # Epochs vs rewards
    num_epochs = len(rewards_history)
    ax1.plot(np.arange(num_epochs) * test_freq, rewards_history, color='grey', alpha=0.3, linestyle='-', label="epoch reward")
    ax1.plot(np.arange(num_epochs) * test_freq, running_rewards_history, color='green', label="moving average reward")
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
