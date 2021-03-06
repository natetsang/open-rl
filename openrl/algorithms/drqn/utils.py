import numpy as np
from typing import Tuple, Union


class ReplayBuffer:
    """
    Initializes a simple FIFO Replay Buffer that can be used for storing transitions and sampling random
    transitions. Once the buffer is full, new transitions will overwrite the oldest transitions in the buffer.
    """
    def __init__(self, state_dims: int, action_dims: int, history_length: int, capacity: int = 1000000,
                 batch_size: int = 64) -> None:
        self.capacity = capacity
        self.batch_size = batch_size

        self.buffer_state = np.empty(shape=(capacity, history_length, state_dims))
        self.buffer_action = np.empty(shape=(capacity, action_dims))
        self.buffer_reward = np.empty(shape=(capacity, 1))
        self.buffer_next_state = np.empty(shape=(capacity, history_length, state_dims))
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
