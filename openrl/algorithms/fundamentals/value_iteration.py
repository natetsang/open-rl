import gym
import gym_walk  # noqa
import numpy as np


def value_iteration(
    P: dict[dict[tuple[float, int, float, bool]]], gamma: float, theta: float
) -> tuple[np.ndarray, dict[int, int]]:
    """Adapted from @source:GDRL chapter 3"""
    num_states = len(P)
    num_actions = len(P[0])
    policy = {}
    delta = 0

    # Initialize V(s), for all s in S, arbitrarily except that V(terminal) = 0
    V = np.zeros(num_states)

    # Run multiple sweeps through state space until convergence
    while True:
        Q = np.zeros((num_states, num_actions))  # Initialize q-values for every sweep

        # loop through each state s in S
        for s in range(num_states):
            # Get all q-values (i.e. action-values) by looping through all actions a
            for a in range(num_actions):
                for prob, next_state, reward, done in P[s][a]:
                    Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))

            # extract the greedy policy w.r.t. the q-values
            policy[s] = np.argmax(Q[s])

        # calculate delta between iterations
        V_new = np.max(Q, axis=1)
        delta = np.max(np.abs(V - V_new))

        # If our estimated V changed less than our threshold, we've found V* and pi*
        if delta < theta:
            break

        V = V_new.copy()

    return V, policy


if __name__ == "__main__":
    env = gym.make("SlipperyWalkFive-v0")
    P = env.env.P  # Get probability matrix

    V, policy = value_iteration(P=P, gamma=1.0, theta=1e-10)

    # Round to make it easier to read
    V_actual = [round(x, 2) for x in V]
    # first and last states are terminal, so prune these
    V_actual = V_actual[1:6]
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