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


def compute_G_forward_view_lambda(rewards: list, V, states, gamma: float = 0.95, lambda_: float = 0.9) -> float:
    num_steps_to_go = len(rewards)
    G_lambda = 0

    # First term
    for n in range(1, num_steps_to_go):
        state_n = states[n]
        lambda_n = lambda_**(n - 1)
        rewards_n = rewards[:n]
        G_n = (1 - lambda_n) * compute_G(rewards=rewards_n, gamma=gamma) * gamma**n * V[state_n]
        G_lambda += G_n

    # Second term
    lambda_T = lambda_**(num_steps_to_go - 1)
    G_T = compute_G(rewards=rewards, gamma=gamma)
    G_lambda += lambda_T * G_T

    return G_lambda


def exponential_decay(param: float, decay_factor: float, min_val: float) -> float:
    if param > min_val:
        param *= decay_factor
        param = max(decay_factor, min_val)
    return param


def forward_view_td_lambda(
    policy: dict[int, int],
    env: gym.Env,
    gamma: float,
    alpha: float,
    lambda_: float,
    num_episodes: int,
) -> np.ndarray:
    # Notice - since we don't have access to P, we sample from env instead
    num_states = env.observation_space.n
    V = np.zeros(num_states)

    for _ in range(num_episodes):
        G = 0
        N = np.zeros(num_states)
        episode = generate_episode(policy, env)

        for t in range(len(episode)):
            states = episode[t:][0]  # (s, a, r, d) tuple
            state = states[0]
            N[state] += 1
            rewards = episode[t:, 2]
            G = compute_G_forward_view_lambda(rewards, V, states, gamma, lambda_)

            # stationary - based on number of visits to state s
            # V[state] = V[state] + (1 / N[state]) * (G - V[state])

            # non-stationary - based on alpha...alpha must decay to guarantee convergence
            V[state] = V[state] + alpha * (G - V[state])
            alpha = exponential_decay(alpha, decay_factor=0.99975, min_val=0.001)
    return V


def backward_view_td_lambda(
    policy: dict[int, int],
    env: gym.Env,
    gamma: float,
    alpha: float,
    lambda_: float,
    num_episodes: int,
    replacing_trace: bool,
) -> np.ndarray:
    num_states = env.observation_space.n
    V = np.zeros(num_states)

    for _ in range(num_episodes):
        E = np.zeros(num_states)
        state = env.reset()
        done = False
        while not done:
            action = policy[state]
            next_state, reward, done, _ = env.step(action)
            target = reward + gamma * V[next_state] * (not done)
            # Update trace
            E[state] = E[state] + 1
            if replacing_trace:
                E = E.clip(0, 1)
            # We update the entire V function
            V = V + alpha * (target - V[state]) * E

            # Decay trace
            E = gamma * lambda_ * E
            alpha = exponential_decay(alpha, decay_factor=0.99975, min_val=0.001)
            state = next_state

    return V


def sarsa_lambda(
    policy: dict[int, int],
    env: gym.Env,
    gamma: float,
    alpha: float,
    lambda_: float,
    epsilon: float,
    num_episodes: int,
    replacing_trace: bool,
) -> np.ndarray:
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    Q = np.zeros((num_states, num_actions))
    policy = {}
    for _ in range(num_episodes):
        E = np.zeros((num_states, num_actions))
        state = env.reset()
        action = (
            policy[state]
            if np.random.random() > epsilon
            else np.random.randint(num_actions)
        )
        epsilon = exponential_decay(epsilon, decay_factor=0.99975, min_val=0.001)
        done = False
        while not done:
            next_state, reward, done, _ = env.step(action)
            next_action = (
                policy[next_state]
                if np.random.random() > epsilon
                else np.random.randint(num_actions)
            )
            target = reward + gamma * Q[next_state][next_action] * (not done)
            # Update trace
            E[state][action] = E[state][action] + 1
            if replacing_trace:
                E = E.clip(0, 1)
            # We update the entire Q function
            Q = Q + alpha * (target - Q[state][action]) * E

            # Decay trace
            E = gamma * lambda_ * E
            alpha = exponential_decay(alpha, decay_factor=0.99975, min_val=0.001)

            # extract policy - policy imporvement w.r.t to visited states
            # this is the only difference between prediction and control
            policy[state] = np.argmax(Q[state])

            state, action = next_state, next_action

    return Q, policy


def watkins_q_learning(
    policy: dict[int, int],
    env: gym.Env,
    gamma: float,
    alpha: float,
    epsilon: float,
    lambda_: float,
    num_episodes: int,
    replacing_trace: bool,
) -> np.ndarray:
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    Q = np.zeros((num_states, num_actions))
    policy = {}
    for _ in range(num_episodes):
        E = np.zeros((num_states, num_actions))
        state = env.reset()
        action = (
            policy[state]
            if np.random.random() > epsilon
            else np.random.randint(num_actions)
        )
        epsilon = exponential_decay(epsilon, decay_factor=0.99975, min_val=0.001)
        done = False
        while not done:
            next_state, reward, done, _ = env.step(action)
            next_action = (
                policy[next_state]
                if np.random.random() > epsilon
                else np.random.randint(num_actions)
            )
            next_action_is_greedy = Q[next_state][next_action] == Q[next_state].max()

            target = reward + gamma * np.max(Q[next_state]) * (not done)
            # Update trace
            E[state][action] = E[state][action] + 1
            if replacing_trace:
                E = E.clip(0, 1)
            # We update the entire Q function
            Q = Q + alpha * (target - Q[state][action]) * E

            # Decay trace: If not greedy, cut traces to zero
            E = gamma * lambda_ * E if next_action_is_greedy else E.fill(0)

            alpha = exponential_decay(alpha, decay_factor=0.99975, min_val=0.001)

            # extract policy - policy imporvement w.r.t to visited states
            # this is the only difference between prediction and control
            policy[state] = np.argmax(Q[state])

            state, action = next_state, next_action

    return Q, policy
