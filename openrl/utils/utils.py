import gym
from typing import Tuple


def configure_env(name: str) -> Tuple[gym.Env, bool, int, int, int]:
    environment = gym.make(name)
    is_discrete_env = type(environment.action_space) == gym.spaces.discrete.Discrete

    if is_discrete_env:
        action_dims = 1
        num_actions = environment.action_space.n
        state_dims = len(environment.observation_space.high)
    else:
        action_dims = environment.action_space.shape[0]
        num_actions = None  # Technically, there's an infinite number
        state_dims = environment.observation_space.shape[0]
    return environment, is_discrete_env, state_dims, action_dims, num_actions
