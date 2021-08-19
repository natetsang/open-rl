import numpy as np


class ReplayBuffer:
    def __init__(self, capacity=1000000, batch_size=256):
        self.capacity = capacity
        self.batch_size = batch_size

        self.buffer_state = np.empty(shape=(capacity, 3))
        self.buffer_action = np.empty(shape=(capacity, 1))
        self.buffer_reward = np.empty(shape=(capacity, 1))
        self.buffer_next_state = np.empty(shape=(capacity, 3))
        self.buffer_done = np.empty(shape=(capacity, 1))

        self.size = 0
        self.idx = 0

    def store_transition(self, transition):
        state, action, reward, next_state = transition
        # This will make sure to overwrite the oldest transition if full
        current_index = self.idx % self.capacity
        self.buffer_state[current_index] = state
        self.buffer_action[current_index] = action
        self.buffer_reward[current_index] = reward
        self.buffer_next_state[current_index] = next_state

        # Increment counters
        if self.size < self.capacity:
            self.size += 1
        self.idx += 1

    def sample(self):
        # We can't sample if we don't have enough transitions
        if self.size < self.batch_size:
            return

        idxs = np.random.choice(self.size, self.batch_size)

        batch_state = self.buffer_state[idxs]
        batch_action = self.buffer_action[idxs]
        batch_reward = self.buffer_reward[idxs]
        batch_next_state = self.buffer_next_state[idxs]

        return (batch_state, batch_action, batch_reward, batch_next_state)

    def __len__(self):
        return self.size
