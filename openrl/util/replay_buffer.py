import numpy as np
from typing import Tuple, List, Union


class ReplayBuffer:
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

    def store_transitions_batch(self, batch: List[Tuple]) -> None:
        states, actions, rewards, next_states, dones = [], [], [], [], []
        for transition in batch:
            state, action, reward, next_state, done = transition
            # TODO >> do we really need all this squeezing and converting to Numpy?
            states.append(np.array(np.squeeze(state)))
            actions.append(action)
            rewards.append(reward)
            next_states.append(np.array(np.squeeze(next_state)))
            dones.append(done)

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
        # TODO >> I should change this so I can update the mean without looking through the whole
        #   buffer every single time.
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
