import gym
import gym_walk  # noqa
import numpy as np


def policy_evaluation(
    policy: dict[int, int],
    P: dict[dict[tuple[float, int, float, bool]]],
    gamma: float = 1.0,
    theta: float = 1e-10,
) -> np.ndarray:
    """Adapted from @source:GDRL chapter 3"""
    num_states = len(P)
    delta = 0

    # Initialize V(s), for all s in S, arbitrarily except that V(terminal) = 0
    V = np.zeros(num_states)

    # Run multiple sweeps through state space until convergence
    while True:
        V_new = np.zeros(num_states)  # initialize current iteration estimates to zero

        # loop through each state s in S
        for s in range(num_states):
            # foreach state, loop through all possible actions pi(a|s)
            a = policy[s]
            for prob, next_state, reward, done in P[s][a]:
                V_new[s] += prob * (reward + gamma * V[next_state] * (not done))

        # calculate delta between iterations
        delta = np.max(np.abs(V - V_new))

        V = V_new.copy()

        # If our estimated V changed less than our threshold, we're done
        if delta < theta:
            break

    # Return our estimated value-function
    return V_new


def policy_improvement(
    V: np.ndarray, P: dict[dict[tuple[float, int, float, bool]]], gamma: float = 1.0
) -> dict[int, int]:
    num_states = len(P)
    num_actions = len(P[0])

    # Initialize policy and q-values
    greedy_policy = {}
    Q = np.zeros((num_states, num_actions))

    # loop through each state s in S
    for s in range(num_states):
        # Get all q-values (i.e. action-values) by looping through all actions a
        for a in range(num_actions):
            for prob, next_state, reward, done in P[s][a]:
                Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))

        # Extract the greedy policy w.r.t. the q-values
        greedy_policy[s] = np.argmax(Q[s])
    return greedy_policy


def policy_iteration(
    P: dict[dict[tuple[float, int, float, bool]]],
    gamma: float = 1.0,
    theta: float = 1e-10,
) -> tuple[np.ndarray, dict[int, int]]:
    # Randomly intialize pi
    policy = {s: np.random.choice(list(P[s].keys())) for s in P}
    prev_policy = policy.copy()

    # Run until convergence
    while True:
        # Run policy evaluation to get updated V_pi
        V = policy_evaluation(policy, P, gamma, theta)

        # Run policy improvement to get updated policy
        policy = policy_improvement(V, P, gamma)

        # Check to make sure the new policy is different
        # If the same, we've found V* and pi*
        if policy == prev_policy:
            break

        prev_policy = policy

    return V, policy


if __name__ == "__main__":
    env = gym.make("SlipperyWalkFive-v0")
    P = env.env.P  # Get probability matrix
    
    V, policy = policy_iteration(P=P, gamma=1.0, theta=1e-10)
    
    V_rounded = [round(x, 2) for x in V]
    V_true = [0.0, 0.67, 0.89, 0.96, 0.99, 1.0, 0.0]
    policy_true = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 0}

    if V_rounded == V_true:
        print("Converged to the correct value function!")
    else:
        print("Did not converge to the correct value function!")
        print("Expected: ", V_true)
        print("Actual: ", V_rounded)
    if policy == policy_true:
        print("Converged to the correct policy!")
    else:
        print("Did not converge to the correct policy!")
        print("Expected: ", policy_true)
        print("Actual: ", policy)
