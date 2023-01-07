import gym
import gym_walk  # noqa
import numpy as np


def generate_episode(
    policy: dict[int, int], env: gym.Env, epsilon: float = 0.0
) -> list[tuple[int, int, float, bool]]:
    num_actions = env.action_space.n
    trajectory = []
    done = False
    state = env.reset()

    while not done:
        # select an action using e-greedy
        action = (
            policy[state]
            if np.random.random() > epsilon
            else np.random.randint(num_actions)
        )
        state, reward, done, _ = env.step(action)
        trajectory.append((state, action, reward, done))
    return trajectory


def computed_discounted_rewards(rewards: list[float], gamma: float = 0.95) -> float:
    """
    Compute the rewards-to-go, which are the cumulative rewards from t=t' to T.
    """
    discounted_reward = 0
    for i, r in enumerate(rewards):
        discounted_reward += gamma**i * r
    return discounted_reward


def exponential_decay(param: float, decay_factor: float, min_val: float) -> float:
    """Exponentially decay `param` by `decay_factor` iff it's above `min_val`."""
    if param > min_val:
        param *= decay_factor
        param = max(param, min_val)
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

    # Arbitrarily initialize V
    V = np.zeros(num_states)

    for _ in range(num_episodes):
        # For each episode, reinitialize N
        N = np.zeros(num_states)

        # Sample an episode trajectory by following pi
        episode = generate_episode(policy, env)

        # Loop through each step in the episode
        for t in range(len(episode)):
            state = episode[t][0]  # (s, a, r, d) tuple

            # If doing the first-visit algorithm, ignore states previously-visited states
            if first_visit and N[state] > 0:
                continue

            N[state] += 1  # Mark state as visited

            # Get list of rewards from t
            rewards = np.array(episode)[:, 2]

            # Compute G, the sum of discounted rewards t:T
            G = computed_discounted_rewards(rewards, gamma)

            # stationary approach - based on number of visits to state s
            # V[state] = V[state] + (1 / N[state]) * (G - V[state])

            # non-stationary approach - based on alpha
            V[state] = V[state] + alpha * (G - V[state])

            # V[state] += sum([x[2] * gamma**i for i, x in enumerate(episode[t:])])

        # NOTE: alpha must decay to guarantee convergence
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

    # Arbitrarily initialize Q, pi
    Q = np.zeros((num_states, num_actions))
    policy = {s: np.random.choice(num_actions) for s in range(num_states)}

    for _ in range(num_episodes):
        # For each episode, reinitialize N
        N = np.zeros((num_states, num_actions))

        # Sample an episode trajectory
        episode = generate_episode(policy, env, epsilon)

        # Loop through each step in the episode
        for t in range(len(episode)):
            state = episode[t][0]  # (s, a, r, d) tuple
            action = episode[t][1]

            # If doing the first-visit algorithm, ignore states previously-visited state-actions
            if first_visit and N[state][action] > 0:
                continue

            N[state][action] += 1  # Mark state-action as visited

            # Get list of rewards from t:T
            rewards = np.array(episode)[:, 2]

            # Compute G, the sum of discounted rewards t:T
            G = computed_discounted_rewards(rewards, gamma)

            # stationary approach - based on number of visits to state-action
            # Q[state][action] = Q[state][action] + (1 / N[state][action]) * (G - Q[state][action])

            # non-stationary approach - based on alpha
            Q[state][action] = Q[state][action] + alpha * (G - Q[state][action])

            # extract policy - policy imporvement w.r.t to visited states
            # this is the only difference between prediction and control
            policy[state] = np.argmax(Q[state])

        policy = {s: np.argmax(Q[s]) for s in range(num_states)}
        # NOTE: alpha must decay to guarantee convergence
        alpha = exponential_decay(alpha, decay_factor=0.99975, min_val=0.001)
        epsilon = exponential_decay(epsilon, decay_factor=0.99795, min_val=0.001)
    return Q, policy


if __name__ == "__main__":
    env = gym.make("SlipperyWalkFive-v0")

    Q, policy = mc_control(
        env=env, gamma=1.0, alpha=0.5, epsilon=0.2, num_episodes=3000, first_visit=True
    )

    # Round to make it easier to read
    V_actual = [round(x, 2) for x in np.max(Q, axis=1)]
    # first and last states are terminal, so prune these
    V_actual = V_actual[1:5]
    policy_actual = {k: v for k, v in policy.items() if k not in [0, 6]}

    V_expected = [0.67, 0.89, 0.96, 0.99, 1.0]
    policy_expected = {1: 1, 2: 1, 3: 1, 4: 1, 5: 1}

    if V_actual == V_expected:
        print("Converged to the correct value function!")
    else:
        print("Did not converge to the correct value function!")
        print("Expected: ", V_expected)
        print("Actual:   ", V_actual)
    if policy_actual == policy_expected:
        print("Converged to the correct policy!")
    else:
        print("Did not converge to the correct policy!")
        print("Expected: ", policy_expected)
        print("Actual:   ", policy_actual)
