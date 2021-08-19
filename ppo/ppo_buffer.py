import numpy as np


class PPOBuffer:
    def __init__(self, state_size, action_size, capacity, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.capacity = capacity
        self.batch_size = batch_size
        self.size = 0
        self.idx = 0
        self._initialize_empty_buffers()

    def _initialize_empty_buffers(self):
        self.buffer_state = np.zeros(shape=(self.capacity, self.state_size))
        self.buffer_action = np.zeros(shape=(self.capacity, self.action_size))
        self.buffer_reward = np.zeros(shape=(self.capacity, 1))
        self.buffer_mask = np.zeros(shape=(self.capacity, 1))

        self.buffer_value = np.zeros(shape=(self.capacity, 1))
        self.buffer_logp = np.zeros(shape=(self.capacity, 1))
        self.buffer_adv = np.zeros(shape=(self.capacity, 1))
        self.buffer_return = np.zeros(shape=(self.capacity, 1))

    def store_transition(self, transition):
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

    def sample(self):
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

    def flush(self):
        self.size = 0
        self.idx = 0
        self._initialize_empty_buffers()

    def __len__(self):
        return self.size
