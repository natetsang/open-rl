from typing import Tuple
import numpy as np

ACTION_TRUE_REWARD_DIST = [0.95, 0.9, 0.3]


def run_greedy(n_arms: int, n_episodes: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # Initialize variables needed in algorithm
    q_values = np.zeros(n_arms)
    counts = np.zeros(n_arms)

    # Initialize other variables for tracking and stats (not necessary)
    q_value_history = np.zeros((n_episodes, n_arms))
    action_history = np.empty(n_episodes)
    reward_history = np.empty(n_episodes)

    for i in range(n_episodes):
        # Step 1: Select action/arm
        action = np.argmax(q_values)  # Find and select greedy action

        # Step 2: Take action and get reward
        reward = np.random.binomial(1, ACTION_TRUE_REWARD_DIST[int(action)])

        # Step 3: Update variables based on action taken and reward received
        counts[action] += 1
        q_values[action] = q_values[action] + (reward - q_values[action]) / counts[action]

        # Accounting
        q_value_history[i] = q_values
        action_history[i] = action
        reward_history[i] = reward

    return q_values, q_value_history, action_history, reward_history


if __name__ == '__main__':
    arms = len(ACTION_TRUE_REWARD_DIST)

    q, q_history, a_history, r_history = run_greedy(n_arms=arms, n_episodes=10)
    print("Q_values:\n", q)
    print("Q_values_history:\n", q_history)
    print("Actions:\n", a_history)
    print("Rewards:\n", r_history)
    print("Cumulative Rewards:\n", np.cumsum(r_history))
    regret = np.max(ACTION_TRUE_REWARD_DIST) - [ACTION_TRUE_REWARD_DIST[int(a)] for a in a_history]
    print("Regret:\n", regret)
    print("Cumulative Regret:\n", np.cumsum(regret))
