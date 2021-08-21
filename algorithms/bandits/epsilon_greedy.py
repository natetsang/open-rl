from typing import Tuple, List
import numpy as np

ACTION_TRUE_REWARD_DIST = [0.95, 0.9, 0.3]
INIT_EPSILON = 1.0


def linear_decay(epsilon: float, min_epsilon: float, decay_rate: float):
    """
    Linearly decay epsilon by subtracting it by `decay_rate`. Epsilon must not
    drop below `min_epsilon`.
    """
    if epsilon > min_epsilon:
        epsilon -= decay_rate
        return np.clip(epsilon, min_epsilon, epsilon)
    return min_epsilon


def exponential_decay(epsilon: float, min_epsilon: float, decay_rate: float):
    """
    Exponentially decay epsilon by multiplying it by (1 - `decay_rate`). Epsilon must not
    drop below `min_epsilon` and `decay_rate` must be within range [0, 1].
    """
    if epsilon > min_epsilon:
        epsilon *= (1 - decay_rate)
        return np.clip(epsilon, min_epsilon, epsilon)
    return min_epsilon


def run_epsilon_greedy(n_arms: int,
                       init_epsilon: float = 1.0,
                       min_epsilon: float = 0.01,
                       decay_strategy: str = None,
                       exp_decay_rate: float = None,
                       n_episodes: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Initialize variables needed in algorithm
    q_values = np.zeros(n_arms)
    counts = np.zeros(n_arms)

    # Initialize other variables for tracking and stats (not necessary)
    q_value_history = np.zeros((n_episodes, n_arms))
    action_history = np.empty(n_episodes)
    reward_history = np.empty(n_episodes)
    epsilon_history = np.empty(n_episodes)

    epsilon = init_epsilon

    # If linear decay, calculate epsilon reduction per episode
    if decay_strategy == 'linear':
        lin_decay_rate = (epsilon - min_epsilon) / (n_episodes - 1)

    for i in range(n_episodes):
        # Step 1: Select action/arm
        # Decay epsilon
        if i > 0:
            if decay_strategy == 'linear':
                epsilon = linear_decay(epsilon=epsilon, min_epsilon=min_epsilon, decay_rate=lin_decay_rate)
            elif decay_strategy in ('exponential', 'exp'):
                epsilon = exponential_decay(epsilon=epsilon, min_epsilon=min_epsilon, decay_rate=exp_decay_rate)

        # Sample random number and compare to epsilon to determine action selection strategy
        if np.random.random() > epsilon:
            # Select greedy action
            action = np.argmax(q_values)
        else:
            # Select random action
            action = np.random.randint(n_arms)

        # Step 2: Take action and get reward
        reward = np.random.binomial(1, ACTION_TRUE_REWARD_DIST[action])

        # Step 3: Update variables based on action taken and reward received
        counts[action] += 1
        q_values[action] = q_values[action] + (reward - q_values[action]) / counts[action]

        # Accounting
        q_value_history[i] = q_values
        action_history[i] = action
        reward_history[i] = reward
        epsilon_history[i] = round(epsilon, 3)

    return q_values, q_value_history, action_history, reward_history, epsilon_history


if __name__ == '__main__':
    arms = len(ACTION_TRUE_REWARD_DIST)

    q, q_history, a_history, r_history, eps_history = run_epsilon_greedy(n_arms=arms,
                                                                         init_epsilon=INIT_EPSILON,
                                                                         min_epsilon=0.01,
                                                                         decay_strategy="exponential",
                                                                         exp_decay_rate=0.2,
                                                                         n_episodes=10)
    print("Q_values:\n", q)
    print("Q_values_history:\n", q_history)
    print("Actions:\n", a_history)
    print("Rewards:\n", r_history)
    print("Cumulative Rewards:\n", np.cumsum(r_history))
    regret = np.max(ACTION_TRUE_REWARD_DIST) - [ACTION_TRUE_REWARD_DIST[int(a)] for a in a_history]
    print("Regret:\n", regret)
    print("Cumulative Regret:\n", np.cumsum(regret))
    print("Epsilons:\n", eps_history)
