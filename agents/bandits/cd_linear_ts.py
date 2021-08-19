"""
This implementation doesn't work.

Inspired by
@https://github.com/ntucllab/striatum/blob/408947981e18f22db308d695c8954112ca02041a/striatum/bandit/linthompsamp.py#L126
"""
import numpy as np
import matplotlib.pyplot as plt


FILENAME = "dataset.txt"
n_arms = 10
context_dim = 100
ALPHA = 0.001
DELTA = 0.61
R = 0.01
EPSILON = 0.71


def process(record):
    values = record.split()  # Fetch individual values to perform operations on them
    values = list(map(int, values))  # Convert all strings in the list to integer

    # Note that data_arms index range from 1 to 10 while policy arms index range from 0 to 9.
    arm = values.pop(0) - 1

    reward = values.pop(0)  # Get reward for the current action
    context = np.asarray(values)  # Create the context array
    return arm, reward, context


def run_lin_ts():
    # Initialize vars - some need to be dicts for assigning numpy linalg arrays
    B = np.identity(context_dim)
    mu_hat = np.zeros((context_dim, 1))
    f = np.zeros((context_dim, 1))
    expected_reward = {}
    ctr = []

    with open(FILENAME, "r") as file:
        # Strip the new line character from the end of each line
        data = [line.strip("\n") for line in file]

    # Iterate over all 10,000 data points
    t = 0
    rewards = []
    cumulative_rewards = 0

    for record in data:
        actual_arm, observed_reward, user_context = process(record)

        v = R * np.sqrt(24 / EPSILON * context_dim * np.log(1 / DELTA))
        mu_tilde = np.random.multivariate_normal(mean=mu_hat.flat, cov=v ** 2 * np.linalg.inv(B))[..., np.newaxis]
        print("MU TILDE", mu_tilde.shape)
        # For each arm, calculate its UCB
        for arm in range(n_arms):
            expected_reward[arm] = np.dot(user_context.T, mu_tilde)

        # Select arm with maximum UCB
        pred_arm = np.argmax(list(expected_reward.values())).item()
        # print("expected_reward", expected_reward)
        # If the algo_arm matches the data_arm, we can use it
        if pred_arm == actual_arm:
            # Update matrix A and B
            B += np.outer(user_context, user_context)
            f += np.dot(np.reshape(user_context, (-1, 1)), observed_reward)
            mu_hat = np.dot(np.linalg.inv(B), f)

            # Increment the time step
            t += 1

            # Update rewards
            cumulative_rewards += observed_reward
            rewards.append(observed_reward)

            # Calculate CTR for current step t
            ctr.append(cumulative_rewards / t)
    return ctr


if __name__ == "__main__":
    ctr_results = run_lin_ts()
    p4 = plt.plot([i for i in range(len(ctr_results))], ctr_results, label='alpha = 0.001', color='m')

    ax = plt.gca()
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels)
    ax.set_xlabel('X axis (time steps) ->')
    ax.set_ylabel('Y axis (CTR) ->')
    ax.set_title('Contextual Bandit Learning for Different alpha values')
    ax.grid(True, linestyle='-.')
    ax.set_yticks(np.arange(0, 1, 0.1))
    plt.show()


