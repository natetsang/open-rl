"""
Example implementation of a simple Bandit Environment
"""


import numpy as np
import random
import gym
from gym.spaces import Discrete, Box
from typing import Tuple


class SimpleContextualBandit(gym.Env):
    def __init__(self):
        self.action_space = Discrete(3)
        self.observation_space = Box(low=-1., high=1., shape=(2, ))
        self.cur_context = None

    def reset(self) -> np.ndarray:
        """ There are two possible states: [-1, 1] and [1, -1]. """
        self.cur_context = random.choice([-1., 1.])
        return np.array([self.cur_context, -self.cur_context])

    def step(self, action: int) -> Tuple[np.ndarray, int, bool, dict]:
        """
        Given the state, do a lookup to find the reward for the action taken.
        Note that although the states are [-1, 1] and [1, -1], we just map this to -1 and 1.
        """
        rewards_for_context = {
            -1.: [-10, 0, 10],
            1.: [10, 0, -10],
        }
        reward = rewards_for_context[self.cur_context][action]
        return (np.array([-self.cur_context, self.cur_context]), reward, True,
                {
                    "regret": 10 - reward
                })



