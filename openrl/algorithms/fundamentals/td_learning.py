import gym
import numpy as np


def exponential_decay(param: float, decay_factor: float, min_val: float) -> float:
    if param > min_val:
        param *= decay_factor
        param = max(decay_factor, min_val)
    return param


def td_prediction(
    policy: dict[int, int],
    env: gym.Env,
    gamma: float,
    alpha: float,
    num_episodes: int,
) -> np.ndarray:
    num_states = env.observation_space.n
    V = np.zeros(num_states)

    for _ in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            action = policy[state]
            next_state, reward, done, _ = env.step(action)
            target = reward + gamma * V[next_state] * (not done)
            V[state] = V[state] + alpha * (target - V[state])

            alpha = exponential_decay(alpha, decay_factor=0.99975, min_val=0.001)
            state = next_state

    return V


def sarsa(
    policy: dict[int, int],
    env: gym.Env,
    gamma: float,
    alpha: float,
    epsilon: float,
    num_episodes: int,
) -> np.ndarray:
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    Q = np.zeros((num_states, num_actions))
    policy = {}
    for _ in range(num_episodes):
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
            Q[state][action] = Q[state][action] + alpha * (target - Q[state][action])

            alpha = exponential_decay(alpha, decay_factor=0.99975, min_val=0.001)

            # extract policy - policy imporvement w.r.t to visited states
            # this is the only difference between prediction and control
            policy[state] = np.argmax(Q[state])

            state, action = next_state, next_action

    return Q, policy


def q_learning(
    policy: dict[int, int],
    env: gym.Env,
    gamma: float,
    alpha: float,
    epsilon: float,
    num_episodes: int,
) -> np.ndarray:
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    Q = np.zeros((num_states, num_actions))
    policy = {}
    for _ in range(num_episodes):
        state = env.reset()
        epsilon = exponential_decay(epsilon, decay_factor=0.99975, min_val=0.001)
        done = False
        while not done:
            action = (
                policy[state]
                if np.random.random() > epsilon
                else np.random.randint(num_actions)
            )
            next_state, reward, done, _ = env.step(action)
            target = reward + gamma * np.max(Q[next_state]) * (not done)
            Q[state][action] = Q[state][action] + alpha * (target - Q[state][action])

            alpha = exponential_decay(alpha, decay_factor=0.99975, min_val=0.001)

            # extract policy - policy imporvement w.r.t to visited states
            # this is the only difference between prediction and control
            policy[state] = np.argmax(Q[state])

            state = next_state

    return Q, policy


def get_action_probs(state, epsilon, Q, num_actions):
    probs = [
        epsilon / (num_actions - 1)
    ] * num_actions  # TODO(ntsang):this might be (epsilon/num_actions)
    greedy_action_idx = np.argmax(Q[state])
    probs[greedy_action_idx] = (
        1.0 - epsilon
    )  # TODO(ntsang): if above, this will be +=, not +
    return np.array(probs)


def expected_sarsa(
    policy: dict[int, int],
    env: gym.Env,
    gamma: float,
    alpha: float,
    epsilon: float,
    num_episodes: int,
) -> np.ndarray:
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    Q = np.zeros((num_states, num_actions))
    policy = {}
    for _ in range(num_episodes):
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
            next_action_probs = get_action_probs(next_state, epsilon, Q, num_actions)
            expected_next_q = np.sum(next_action_probs * Q[next_state])
            target = reward + gamma * expected_next_q * (not done)
            Q[state][action] = Q[state][action] + alpha * (target - Q[state][action])

            alpha = exponential_decay(alpha, decay_factor=0.99975, min_val=0.001)

            # extract policy - policy imporvement w.r.t to visited states
            # this is the only difference between prediction and control
            policy[state] = np.argmax(Q[state])

            state = next_state

    return Q, policy


def double_q_learning(
    policy: dict[int, int],
    env: gym.Env,
    gamma: float,
    alpha: float,
    epsilon: float,
    num_episodes: int,
) -> np.ndarray:
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    Q = np.zeros((num_states, num_actions))
    Q1 = np.zeros((num_states, num_actions))
    Q2 = np.zeros((num_states, num_actions))
    policy = {}
    for _ in range(num_episodes):
        state = env.reset()
        epsilon = exponential_decay(epsilon, decay_factor=0.99975, min_val=0.001)
        done = False
        while not done:
            action = (
                policy[state]
                if np.random.random() > epsilon
                else np.random.randint(num_actions)
            )
            next_state, reward, done, _ = env.step(action)
            if np.random.random() < 0.5:
                target = reward + gamma * Q2[next_state][np.argmax(Q1[next_state])] * (
                    not done
                )
                Q1[state][action] = Q1[state][action] + alpha * (
                    target - Q1[state][action]
                )
            else:
                target = reward + gamma * Q1[next_state][np.argmax(Q2[next_state])] * (
                    not done
                )
                Q2[state][action] = Q2[state][action] + alpha * (
                    target - Q2[state][action]
                )

            alpha = exponential_decay(alpha, decay_factor=0.99975, min_val=0.001)

            # extract policy - policy imporvement w.r.t to visited states
            # this is the only difference between prediction and control
            # Update Q, which is the average
            Q[state][action] = np.mean([Q1[state][action], Q2[state][action]])
            policy[state] = np.argmax(Q[state])
            state = next_state

    return Q, policy


def double_sarsa(
    policy: dict[int, int],
    env: gym.Env,
    gamma: float,
    alpha: float,
    epsilon: float,
    num_episodes: int,
) -> np.ndarray:
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    Q = np.zeros((num_states, num_actions))
    Q1 = np.zeros((num_states, num_actions))
    Q2 = np.zeros((num_states, num_actions))
    policy = {}
    for _ in range(num_episodes):
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
            if np.random.random() < 0.5:
                target = reward + gamma * Q2[next_state][next_action] * (
                    not done
                )
                Q1[state][action] = Q1[state][action] + alpha * (
                    target - Q1[state][action]
                )
            else:
                target = reward + gamma * Q1[next_state][next_action] * (
                    not done
                )
                Q2[state][action] = Q2[state][action] + alpha * (
                    target - Q2[state][action]
                )

            alpha = exponential_decay(alpha, decay_factor=0.99975, min_val=0.001)

            # extract policy - policy imporvement w.r.t to visited states
            # this is the only difference between prediction and control
            # Update Q, which is the average
            Q[state][action] = np.mean([Q1[state][action], Q2[state][action]])
            policy[state] = np.argmax(Q[state])
            state = next_state

    return Q, policy
