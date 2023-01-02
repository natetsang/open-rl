import gym
import gym_walk
import numpy as np


def value_iteration(
    P: dict, gamma: float, theta: float
) -> tuple[np.ndarray, dict[int, int]]:
    """Inspired by @source: GDRL chapter 3"""
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
            # Get all q-values
            for a in range(num_actions):
                for prob, next_state, reward, done in P[s][a]:
                    Q[s][a] += prob * (reward + gamma * V[next_state] * (not done))

            # Extract the greedy policy w.r.t. the q-values
            policy[s] = np.argmax(Q[s])

        # calculate delta
        V_new = np.max(Q, axis=1)
        delta = np.max(np.abs(V - V_new))

        # update V
        V = V_new.copy()

        if delta < theta:
            break

    return V, policy


if __name__ == "__main__":
    env = gym.make('SlipperyWalkFive-v0')
    gamma = 1.0
    theta = 1e-10
    P = env.env.P
    V, policy = value_iteration(P=P, gamma=gamma, theta=theta)
    print(V)
