import numpy as np
import matplotlib.pyplot as plt


FILENAME = "dataset.txt"
n_arms = 10
context_dim = 100
ALPHA = 0.001


def process(record):
    values = record.split()  # Fetch individual values to perform operations on them
    values = list(map(int, values))  # Convert all strings in the list to integer

    # Note that data_arms index range from 1 to 10 while policy arms index range from 0 to 9.
    arm = values.pop(0) - 1

    reward = values.pop(0)  # Get reward for the current action
    context = np.asarray(values)  # Create the context array
    return arm, reward, context


def run_lin_ucb():
    # Initialize vars - some need to be dicts for assigning numpy linalg arrays
    theta = {}
    A = {}
    b = {}
    ucb = {}
    ctr = []

    for i in range(n_arms):
        A[i] = np.identity(context_dim)
        b[i] = np.zeros((context_dim, 1))
        ucb[i] = 0


    with open(FILENAME, "r") as file:
        # Strip the new line character from the end of each line
        data = [line.strip("\n") for line in file]

    # Iterate over all 10,000 data points
    t = 0
    rewards = []
    cumulative_rewards = 0
    for record in data:
        actual_arm, observed_reward, user_context = process(record)

        # For each arm, calculate its UCB
        for arm in range(n_arms):
            theta[arm] = np.dot(np.linalg.inv(A[arm]), b[arm])
            ucb[arm] = np.dot(theta[arm].T, user_context) + \
                       np.dot(ALPHA, np.sqrt(np.dot(user_context.T, np.dot(np.linalg.inv(A[arm]), user_context))))

        # Select arm with maximum UCB
        pred_arm = np.argmax(list(ucb.values())).item()

        # If the algo_arm matches the data_arm, we can use it
        if pred_arm == actual_arm:
            # Update matrix A and B
            A[pred_arm] += np.outer(user_context, user_context)
            b[pred_arm] += observed_reward * np.reshape(user_context, (context_dim, 1))

            # Increment the time step
            t += 1

            # Update rewards
            cumulative_rewards += observed_reward
            rewards.append(observed_reward)

            # Calculate CTR for current step t
            ctr.append(cumulative_rewards / t)
    return ctr


if __name__ == "__main__":
    ctr_results = run_lin_ucb()
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


