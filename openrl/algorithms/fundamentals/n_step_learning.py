import gym
import numpy as np


def exponential_decay(param: float, decay_factor: float, min_val: float) -> float:
    if param > min_val:
        param *= decay_factor
        param = max(decay_factor, min_val)
    return param


def discount_rewards(rewards: list, gamma: float = 0.95) -> float:
    """
    Compute the rewards-to-go, which are the cumulative rewards from t=t' to T.
    """
    discounted_rewards = []
    total_ret = 0
    for r in rewards[::-1]:
        # Without discount
        # total_ret = r + total_ret

        # With discount
        total_ret = r + gamma * total_ret
        discounted_rewards.insert(0, total_ret)
    return np.sum(discounted_rewards)


def nstep_prediction(
    policy: dict[int, int],
    env: gym.Env,
    gamma: float,
    alpha: float,
    n_steps: int,
    num_episodes: int,
) -> np.ndarray:
    num_states = env.observation_space.n
    V = np.zeros(num_states)

    for _ in range(num_episodes):
        state = env.reset()
        done = False
        trajectory = []
        while True:
            trajectory = trajectory[1:]  # remove first element

            # Gather and append steps
            while len(trajectory) < n_steps and not done:
                action = policy[state]
                next_state, reward, done, _ = env.step(action)
                trajectory.append((state, action, reward, done))
                state = next_state

            state_to_update = trajectory[0][0]  # get the first state in the n-step traj
            n_rewards = np.array(trajectory)[:, 2]  # get all rewards, r(t=0), r(t=1), ... r(t=n-1)
            n_discounted_rewards: float = discount_rewards(n_rewards, gamma)
            target = n_discounted_rewards + gamma**(len(n_discounted_rewards) - 1) * V[next_state] * (not done)
            V[state_to_update] = V[state_to_update] + alpha * (target - V[state_to_update])

            # TODO(ntsang): this is slightly different than GDRL
            # I think the 'and done' is unnecessary here
            if len(trajectory) == 0 and done:
                break
        alpha = exponential_decay(alpha, decay_factor=0.99975, min_val=0.001)  # seems out of place, perhaps remove?
    return V
