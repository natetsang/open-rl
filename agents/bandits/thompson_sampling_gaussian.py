from typing import Tuple, List, Union
import numpy as np
import tensorflow_probability as tfp

TRUE_MU = [6, 4, 3]  # actual mean
TRUE_VARIANCE = [2, 2, 1]  # actual variance
tau = 1 / np.array(TRUE_VARIANCE)  # actual precision


def sample_from_true_distribution(action: int, num_samples: int = None) -> Union[float, np.ndarray]:
    std = np.sqrt(1 / tau[action])
    return np.random.normal(loc=TRUE_MU[action], scale=std, size=num_samples)


def run_thompson_sampling_gaussian(n_arms: int,
                                   n_samples_per_episode: int = 1,
                                   n_episodes: int = 10) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray,
                                                                  List[tfp.distributions.Distribution]]:
    # Initialize variables needed in algorithm
    q_values = np.zeros(n_arms)
    counts = np.zeros(n_arms)

    tau_0 = np.zeros(n_arms)  # posterior precision
    mu_0 = np.zeros(n_arms)  # posterior mean

    std_0 = np.sqrt(1 / tau_0)
    priors = [tfp.distributions.Normal(loc=mu_0[i], scale=std_0[i]) for i in range(n_arms)]  # Initialize priors

    # Initialize other variables for tracking and stats (not necessary)
    q_value_history = np.zeros((n_episodes, n_arms))
    action_history = np.empty(n_episodes)
    reward_history = np.empty(n_episodes)

    for i in range(n_episodes):
        # Step 1: Select action/arm based on observed rewards from previous episodes
        Rs = np.zeros(n_arms)
        for arm in range(n_arms):
            # For each arm sample k reward distributions R(a) and compute mean reward
            prior = priors[arm]
            R_a = np.mean(prior.sample(n_samples_per_episode))
            Rs[arm] = R_a

        action = np.argmax(Rs)  # Select greedy action from the sampled estimates

        # Step 2: Take action and get reward
        reward = sample_from_true_distribution(int(action))

        # Step 3: Update variables based on action taken and reward received
        counts[action] += 1
        q_values[action] = q_values[action] + (reward - q_values[action]) / counts[action]

        # Step 3b: Update posterior based on action taken and reward received
        # Update mean and precision of posterior
        mu_0[action] = ((tau_0[action] * mu_0[action]) + (tau[action] * counts[action] * q_values[action])) / \
                       (tau_0[action] + counts[action] * tau[action])
        tau_0[action] = tau_0[action] + counts[action] * tau[action]

        posterior = tfp.distributions.Normal(loc=mu_0[action], scale=np.sqrt(1 / tau_0[action]))

        # This is another implementation that works, is a bit simpler, and you don't have to keep track of tau or mu!
        # @source https://github.com/mimoralea/gdrl/blob/master/notebooks/chapter_04/chapter-04.ipynb
        # posterior = tfp.distributions.Normal(loc=q_values[action], scale=np.sqrt(1 / counts[action]))

        # Now the posterior becomes the prior for the next update
        priors[action] = posterior

        # Accounting
        q_value_history[i] = q_values
        action_history[i] = action
        reward_history[i] = reward

    return q_values, q_value_history, action_history, reward_history, priors


if __name__ == '__main__':
    arms = len(TRUE_MU)

    q, q_history, a_history, r_history, dist = run_thompson_sampling_gaussian(n_arms=arms,
                                                                              n_samples_per_episode=1,
                                                                              n_episodes=10)
    print("Q_values:\n", q)
    print("Q_values_history:\n", q_history)
    print("Actions:\n", a_history)
    print("Rewards:\n", r_history)
    print("Cumulative Rewards:\n", np.cumsum(r_history))
    regret = np.max(TRUE_MU) - [TRUE_MU[int(a)] for a in a_history]
    print("Regret:\n", regret)
    print("Cumulative Regret:\n", np.cumsum(regret))
    print("Final posterior:\n", [f"Arm {arm}: (mean={dist[arm].loc}; std={dist[arm].scale})"
                                 for arm in range(arms)])

