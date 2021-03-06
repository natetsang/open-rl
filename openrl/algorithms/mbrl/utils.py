import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from typing import List, Tuple, Union
import copy


def add_noise(input_data, noise_to_signal: int = 0.01) -> np.ndarray:
    """
    @source https://github.com/berkeleydeeprlcourse/homework_fall2020/blob/master/hw4/cs285/infrastructure/utils.
    """
    input_data = np.array(input_data)
    noised_data = copy.deepcopy(input_data)  # (num data points, dim)

    # Mean of data
    mean_data = np.mean(noised_data, axis=0)

    # If mean is 0, make it 0.001 to avoid 0 issues later for dividing by std
    mean_data[mean_data == 0] = 0.000001

    # Width of normal distribution to sample noise from larger magnitude number = could have larger magnitude noise
    std_of_noise = mean_data * noise_to_signal
    for j in range(mean_data.shape[0]):
        noised_data[:, j] = np.copy(noised_data[:, j] + np.random.normal(
            0, np.absolute(std_of_noise[j]), (noised_data.shape[0],)))

    return noised_data


class ReplayBufferWithNoise:
    """
    Initializes a simple FIFO Replay Buffer that can be used for storing transitions and sampling random
    transitions. Once the buffer is full, new transitions will overwrite the oldest transitions in the buffer.
    """
    def __init__(self, state_dims: int, action_dims: int, batch_size: int = 64, capacity: int = 1000000) -> None:
        self.capacity = capacity
        self.batch_size = batch_size

        self.buffer_state = np.empty(shape=(capacity, state_dims))
        self.buffer_action = np.empty(shape=(capacity, action_dims))
        self.buffer_reward = np.empty(shape=(capacity, 1))
        self.buffer_next_state = np.empty(shape=(capacity, state_dims))
        self.buffer_done = np.empty(shape=(capacity, 1))

        self.data_statistics = None
        self._size = 0
        self._idx = 0

    def store_transition(self, transition: Tuple) -> None:
        """
        This stores a transition in the buffer. There transition is a
        tuple of (state, action, reward, next_state, done).

        If we've reached capacity, we begin to overwrite the oldest transitions.
        """
        state, action, reward, next_state, done = transition
        # This will make sure to overwrite the oldest transition if full
        current_index = self._idx % self.capacity

        # Store transition in buffer
        self.buffer_state[current_index] = state
        self.buffer_action[current_index] = action
        self.buffer_reward[current_index] = reward
        self.buffer_next_state[current_index] = next_state
        self.buffer_done[current_index] = done

        # Increment counters
        if self._size < self.capacity:
            self._size += 1
        self._idx += 1

    def store_transitions_batch(self, batch: List[Tuple], noised: bool = False) -> None:
        """
        Store a list of transitions in the replay buffer and update the buffer's statistics!
        :param batch: The list of (s,a,r,s',d) transitions to store
        :param noised: Boolean to indicate whether to add noise to the states and next_states
        :return:
        """
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for transition in batch:
            state, action, reward, next_state, done = transition
            states.append(np.array(tf.squeeze(state)))
            actions.append(action)
            rewards.append(reward)
            next_states.append(np.array(tf.squeeze(next_state)))
            dones.append(done)

        if noised:
            states = add_noise(states)
            next_states = add_noise(next_states)

        transitions = zip(states, actions, rewards, next_states, dones)
        for t in transitions:
            self.store_transition(t)

        # Update the statistics in the replay buffer!
        self.update_data_statistics()

    def sample(self, batch_size: int = None) -> Union[Tuple, None]:
        """
        Returns a batch_size of random transitions from the buffer. If there are less
        than batch_size transitions in the buffer, this returns None. Note that sampling
        does not remove the transitions from the buffer so they could be sampled again.
        """
        if batch_size is None:
            batch_size = self.batch_size

        # We can't sample if we don't have enough transitions
        if self._size < batch_size:
            return

        idxs = np.random.choice(self._size, batch_size)

        batch_state = self.buffer_state[idxs]
        batch_action = self.buffer_action[idxs]
        batch_reward = self.buffer_reward[idxs]
        batch_next_state = self.buffer_next_state[idxs]
        batch_done = self.buffer_done[idxs]

        return batch_state, batch_action, batch_reward, batch_next_state, batch_done

    def all(self) -> Union[Tuple, None]:
        """Return all data"""
        return (self.buffer_state[:self._size],
                self.buffer_action[:self._size],
                self.buffer_reward[:self._size],
                self.buffer_next_state[:self._size],
                self.buffer_done[:self._size])

    def update_data_statistics(self) -> Union[dict, None]:
        if self._size == 0:
            return None

        data_statistics = dict()
        data_statistics['obs_mean'] = np.mean(self.buffer_state)
        data_statistics['obs_std'] = np.std(self.buffer_state)
        data_statistics['acs_mean'] = np.mean(self.buffer_action)
        data_statistics['acs_std'] = np.std(self.buffer_action)
        data_statistics['delta_mean'] = np.mean(self.buffer_next_state - self.buffer_state)
        data_statistics['delta_std'] = np.std(self.buffer_next_state - self.buffer_state)
        self.data_statistics = data_statistics
        return data_statistics

    def __len__(self) -> int:
        return self._size


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
