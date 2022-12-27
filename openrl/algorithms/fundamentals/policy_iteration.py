import numpy as np


def policy_evaluation(policy, P, gamma: float, theta: float):
    """
    Policy evaluation is an algorithm for estimating the state-value function V(s) given a policy pi(s).
    It's referred to as the prediction problem, because wer're predicting a value of a state from the policy.
    @source: GDRL chapter 3
    """
    num_states = len(P)
    delta = 0

    # Initialize V(s), for all s in S, arbitrarily except that V(terminal) = 0
    V = np.zeros(num_states)

    while delta <= theta:
        V_new = np.zeros(num_states)  # initialize current iteration estimates to zero

        # loop through each state s in S
        for s in range(num_states):
            # foreach state, loop through all possible actions pi(a|s)
            a = policy[s]
            for prob, next_state, reward, done in P[s][a]:
                V_new[s] += prob * (reward + gamma * V[next_state] * (not done))

        # calculate delta
        delta = np.abs(V - V_new)

        V = V_new.copy()

    return V_new


def policy_improvement(V, P, gamma=1.0):
    num_states = len(P)
    num_actions = len(P[0])

    # Initialize policy and q-values
    greedy_policy = {}
    Q = np.zeros((num_states, num_actions))

    for s in range(num_states):
        # Get all q-values
        for a in range(num_actions):
            for prob, next_state, reward, done in P[s][a]:
                Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))

        # Extract the greedy policy w.r.t. the q-values
        greedy_policy[s] = np.argmax(Q[s])
    return greedy_policy


def policy_iteration(P, gamma, theta):
    # Randomly intialize pi
    policy = {s: np.random.choice(a) for s, a in enumerate(P)}
    prev_policy = policy.copy()

    while True:
        # Run policy evaluation to get updated V_pi
        V = policy_evaluation(policy, P, gamma, theta)

        # Run policy improvement to get updated policy
        policy = policy_improvement(V, P, gamma)

        # Check to make sure the new policy is different
        # If the same, we've found V* and pi*
        if policy == prev_policy:
            break

    return V, policy
