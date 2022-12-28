import gym
import numpy as np


def generate_episode(
    policy: dict[int, int], env: gym.Env, epsilon: float = 0.0
) -> list[tuple[int, int, float, bool]]:
    done = False
    num_actions = env.action_space.n
    trajectory = []
    state = env.reset()

    while not done:
        action = (
            policy[state]
            if np.random.random() > epsilon
            else np.random.randint(num_actions)
        )
        state, reward, done, _ = env.step(action)
        trajectory.append((state, action, reward, done))
    return trajectory


def compute_G(rewards: list, gamma: float = 0.95) -> float:
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


def exponential_decay(param: float, decay_factor: float, min_val: float) -> float:
    if param > min_val:
        param *= decay_factor
        param = max(decay_factor, min_val)
    return param


def mc_prediction(
    policy: dict[int, int],
    env: gym.Env,
    gamma: float,
    alpha: float,
    num_episodes: int,
    first_visit: bool,
) -> np.ndarray:
    # Notice - since we don't have access to P, we sample from env instead
    num_states = env.observation_space.n
    V = np.zeros(num_states)

    for _ in range(num_episodes):
        G = 0
        N = np.zeros(num_states)
        episode = generate_episode(policy, env)

        for t in range(len(episode)):
            state = episode[t][0]  # (s, a, r, d) tuple
            if first_visit and N[state] > 0:
                continue
            N[state] += 1
            rewards = episode[t:, 2]
            G = compute_G(rewards, gamma)

            # stationary - based on number of visits to state s
            # V[state] = V[state] + (1 / N[state]) * (G - V[state])

            # non-stationary - based on alpha...alpha must decay to guarantee convergence
            V[state] = V[state] + alpha * (G - V[state])
            alpha = exponential_decay(alpha, decay_factor=0.99975, min_val=0.001)
    return V


def mc_control(
    env: gym.Env,
    gamma: float,
    alpha: float,
    epsilon: float,
    num_episodes: int,
    first_visit: bool,
) -> np.ndarray:
    # Notice - since we don't have access to P, we sample from env instead
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    Q = np.zeros((num_states, num_actions))
    policy = {}
    for _ in range(num_episodes):
        G = 0
        N = np.zeros((num_states, num_actions))
        episode = generate_episode(policy, env, epsilon)
        epsilon = exponential_decay(epsilon, decay_factor=0.99975, min_val=0.001)

        for t in range(len(episode)):
            state = episode[t][0]  # (s, a, r, d) tuple
            action = episode[t][1]
            if first_visit and N[state][action] > 0:
                continue
            N[state][action] += 1
            rewards = episode[t:, 2]
            G = compute_G(rewards, gamma)

            # stationary - based on number of visits to state s
            # Q[state][action] = Q[state][action] + (1 / N[state][action]) * (G - Q[state])

            # non-stationary - based on alpha...alpha must decay to guarantee convergence
            Q[state][action] = Q[state][action] + alpha * (G - Q[state][action])
            alpha = exponential_decay(alpha, decay_factor=0.99975, min_val=0.001)

            # extract policy - policy imporvement w.r.t to visited states
            # this is the only difference between prediction and control
            policy = np.argmax(Q[state])

    return Q, policy
