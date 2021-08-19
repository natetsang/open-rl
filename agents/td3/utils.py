import matplotlib.pyplot as plt
import numpy as np
from typing import List, Tuple, Union


class ReplayBuffer:
    """
    Initializes a simple FIFO Replay Buffer that can be used for storing transitions and sampling random
    transitions. Once the buffer is full, new transitions will overwrite the oldest transitions in the buffer.
    """
    def __init__(self, capacity: int = 1000000, batch_size: int = 64) -> None:
        self.capacity = capacity
        self.batch_size = batch_size

        self.buffer_state = np.empty(shape=(capacity, 3))
        self.buffer_action = np.empty(shape=(capacity, 1))
        self.buffer_reward = np.empty(shape=(capacity, 1))
        self.buffer_next_state = np.empty(shape=(capacity, 3))
        self.buffer_done = np.empty(shape=(capacity, 1))

        self.size = 0
        self.idx = 0

    def store_transition(self, transition: Tuple) -> None:
        """
        This stores a transition in the buffer. There transition is a
        tuple of (state, action, reward, next_state, done).

        If we've reached capacity, we begin to overwrite the oldest transitions.
        """
        state, action, reward, next_state, done = transition
        # This will make sure to overwrite the oldest transition if full
        current_index = self.idx % self.capacity

        # Store transition in buffer
        self.buffer_state[current_index] = state
        self.buffer_action[current_index] = action
        self.buffer_reward[current_index] = reward
        self.buffer_next_state[current_index] = next_state
        self.buffer_done[current_index] = done

        # Increment counters
        if self.size < self.capacity:
            self.size += 1
        self.idx += 1

    def sample(self) -> Union[Tuple, None]:
        """
        Returns a batch_size of random transitions from the buffer. If there are less
        than batch_size transitions in the buffer, this returns None. Note that sampling
        does not remove the transitions from the buffer so they could be sampled again.
        """
        # We can't sample if we don't have enough transitions
        if self.size < self.batch_size:
            return

        idxs = np.random.choice(self.size, self.batch_size)

        batch_state = self.buffer_state[idxs]
        batch_action = self.buffer_action[idxs]
        batch_reward = self.buffer_reward[idxs]
        batch_next_state = self.buffer_next_state[idxs]
        batch_done = self.buffer_done[idxs]

        return batch_state, batch_action, batch_reward, batch_next_state, batch_done

    def __len__(self) -> int:
        return self.size


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
