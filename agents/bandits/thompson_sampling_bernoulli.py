from typing import Tuple, List
import numpy as np
import tensorflow_probability as tfp

ACTION_TRUE_REWARD_DIST = [0.95, 0.9, 0.3]


def run_thompson_sampling_bernoulli(n_arms: int,
                                    n_samples_per_episode: int = 1,
                                    n_episodes: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                                                   np.ndarray, List[tfp.distributions.Distribution]]:
    # Initialize variables needed in algorithm
    q_values = np.zeros(n_arms)
    counts = np.zeros(n_arms)

    Rs = np.zeros(n_arms)
    priors = [tfp.distributions.Beta(1, 1) for _ in range(n_arms)]  # Initialize priors for each arm

    # Initialize other variables for tracking and stats (not necessary)
    q_value_history = np.zeros((n_episodes, n_arms))
    action_history = np.empty(n_episodes)
    reward_history = np.empty(n_episodes)
    Rs_history = np.zeros((n_episodes, n_arms))  # History of expected rewards for sampling

    for i in range(n_episodes):
        # Step 1: Select action/arm based on observed rewards from previous episodes
        for arm in range(n_arms):
            # For each arm sample k reward distributions R(a) and compute mean reward
            prior = priors[arm]
            R_a = np.mean(prior.sample(n_samples_per_episode))
            Rs[arm] = R_a

        action = np.argmax(Rs)  # Select greedy action from the sampled estimates

        # Step 2: Take action and get reward
        reward = np.random.binomial(1, ACTION_TRUE_REWARD_DIST[int(action)])

        # Step 3: Update variables based on action taken and reward received
        counts[action] += 1
        q_values[action] = q_values[action] + (reward - q_values[action]) / counts[action]

        # Step 3b: Update posterior based on action taken and reward received
        prior = priors[action]
        alpha = prior.concentration1 + reward  # if reward = 1, increment alpha
        beta = prior.concentration0 + (1 - reward)  # if reward = 0, increment beta

        posterior = tfp.distributions.Beta(alpha, beta)

        # Now the posterior becomes the prior for the next update
        priors[action] = posterior

        # Accounting
        q_value_history[i] = q_values
        action_history[i] = action
        reward_history[i] = reward
        Rs_history[i] = Rs

    return q_values, q_value_history, Rs_history, action_history, reward_history, priors


if __name__ == '__main__':
    arms = len(ACTION_TRUE_REWARD_DIST)

    q, q_history, rs_history, a_history, r_history, dist = run_thompson_sampling_bernoulli(n_arms=arms,
                                                                                           n_samples_per_episode=1,
                                                                                           n_episodes=10)
    print("Q_values:\n", q)
    print("Q_values_history:\n", q_history)
    print("Sampled expected reward, R(a):\n", rs_history)
    print("Actions:\n", a_history)
    print("Rewards:\n", r_history)
    print("Cumulative Rewards:\n", np.cumsum(r_history))
    regret = np.max(ACTION_TRUE_REWARD_DIST) - [ACTION_TRUE_REWARD_DIST[int(a)] for a in a_history]
    print("Regret:\n", regret)
    print("Cumulative Regret:\n", np.cumsum(regret))
    print("Final posterior:\n", [f"Arm {arm}: (alpha={dist[arm].concentration1}; beta={dist[arm].concentration0})"
                                 for arm in range(arms)])

