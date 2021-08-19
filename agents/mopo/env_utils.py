import numpy as np
import gym


## Termination Functions
def _pendulum_termination_fn(obs: np.ndarray) -> np.ndarray:
    return np.zeros(shape=obs.shape[0])


def termination_fn(env: gym.Env, obs: np.ndarray, act: np.ndarray, next_obs: np.ndarray) -> np.ndarray:
    env_name = env.spec.id
    if env_name == "Pendulum-v0":
        return _pendulum_termination_fn(obs)


## Reward Functions
def _pendulum_reward_fn(env: gym.Env, obs: np.ndarray, act: np.ndarray) -> np.ndarray:
    def angle_normalize(x):
        return (((x + np.pi) % (2 * np.pi)) - np.pi)

    th = np.arccos(obs[:, 0])
    thdot = obs[:, 2]

    u = np.clip(act, -env.max_torque, env.max_torque)
    u = np.squeeze(u)
    costs = angle_normalize(th) ** 2 + .1 * thdot ** 2 + .001 * (u ** 2)
    return -costs


def reward_fn(env: gym.Env, obs: np.ndarray, act: np.ndarray, next_obs: np.ndarray) -> np.ndarray:
    env_name = env.spec.id
    if env_name == "Pendulum-v0":
        return _pendulum_reward_fn(env, obs, act)
