"Credit to GDRL Chapter 3"

import numpy as np


def policy_evaluation(policy, P, gamma: float, theta: float):
    """
    Policy evaluation is an algorithm for estimating the state-value function V(s) given a policy pi(s).
    It's referred to as the prediction problem, because wer're predicting a value of a state from the policy
    """
    # Initialize V(s), for all s in S, arbitrarily except that V(terminal) = 0
    V = np.zeros_like(P)

    # Initialize delta = 0
    delta = 0

    while delta <= theta:
        V_new = np.zeros_like(P)  # initialize current iteration estimates to zero
        # Loop through each state s in S
        for s in range(len(P)):
            # foreach state, loop through all possible actions pi(a|s)
            for prob, next_state, reward, done in P[s][policy(s)]:
                V_new[s] += prob * (reward + gamma * V[next_state] * (not done))

        delta = np.abs(V - V_new)
        V = V_new.copy()

    return V_new


def policy_improvement(V, P, gamma=1.0):
    greedy_policy = {}
    Q = np.zeros((len(P), len(P[0])))
    for s in range(len(P)):
        # Get all q-values
        for a in range(len(P[s])):
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
        # Run policy evaluation
        V = policy_evaluation(policy, P, gamma, theta)

        # Run policy improvement
        policy = policy_improvement(V, P, gamma)

        # Check to make sure the new policy is different
        if policy == prev_policy:
            break

    return V, policy
