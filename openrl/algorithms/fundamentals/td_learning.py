import gym
import gym_walk  # noqa
import numpy as np


def exponential_decay(param: float, decay_factor: float, min_val: float) -> float:
    if param > min_val:
        param *= decay_factor
        param = max(param, min_val)
    return param


def td_prediction(
    policy: dict[int, int],
    env: gym.Env,
    gamma: float,
    alpha: float,
    num_episodes: int,
) -> np.ndarray:
    # Notice - since we don't have access to P, we sample from env instead
    num_states = env.observation_space.n
    V = np.zeros(num_states)

    for _ in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            # Sample a step
            action = policy[state]
            next_state, reward, done, _ = env.step(action)
            # Calculate target
            target = reward + gamma * V[next_state] * (not done)
            # Update value estimate
            V[state] = V[state] + alpha * (target - V[state])

            state = next_state

        # Decay parameters
        alpha = exponential_decay(alpha, decay_factor=0.99975, min_val=0.001)
    return V


def sarsa(
    env: gym.Env,
    gamma: float,
    alpha: float,
    epsilon: float,
    num_episodes: int,
) -> np.ndarray:
    # Notice - since we don't have access to P, we sample from env instead
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    # Arbitrarily initialize Q and pi
    Q = np.zeros((num_states, num_actions))
    policy = {s: np.random.choice(num_actions) for s in range(num_states)}

    for _ in range(num_episodes):
        state = env.reset()
        action = (
            policy[state]
            if np.random.random() > epsilon
            else np.random.randint(num_actions)
        )
        done = False
        while not done:
            # Sample a step and select next action
            next_state, reward, done, _ = env.step(action)
            next_action = (
                policy[next_state]
                if np.random.random() > epsilon
                else np.random.randint(num_actions)
            )

            # Calculate the target
            target = reward + gamma * Q[next_state][next_action] * (not done)

            # Update estimated Q
            Q[state][action] = Q[state][action] + alpha * (target - Q[state][action])

            # extract policy - policy imporvement w.r.t to visited states
            # this is the only difference between prediction and control
            policy[state] = np.argmax(Q[state])

            state, action = next_state, next_action

        # Decay parameters
        alpha = exponential_decay(alpha, decay_factor=0.99975, min_val=0.001)
        epsilon = exponential_decay(epsilon, decay_factor=0.99975, min_val=0.001)
    return Q, policy


def q_learning(
    env: gym.Env,
    gamma: float,
    alpha: float,
    epsilon: float,
    num_episodes: int,
) -> np.ndarray:
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    # Arbitrarily initialize Q and pi
    Q = np.zeros((num_states, num_actions))
    policy = {s: np.random.choice(num_actions) for s in range(num_states)}

    for _ in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            # Sample a step
            action = (
                policy[state]
                if np.random.random() > epsilon
                else np.random.randint(num_actions)
            )
            next_state, reward, done, _ = env.step(action)

            # Calculate the target
            target = reward + gamma * np.max(Q[next_state]) * (not done)

            # Update estimated Q
            Q[state][action] = Q[state][action] + alpha * (target - Q[state][action])

            # extract policy - policy imporvement w.r.t to visited states
            # this is the only difference between prediction and control
            policy[state] = np.argmax(Q[state])

            state = next_state

        # Decay parameters
        alpha = exponential_decay(alpha, decay_factor=0.99975, min_val=0.001)
        epsilon = exponential_decay(epsilon, decay_factor=0.99975, min_val=0.001)
    return Q, policy


def get_action_probs(
    state: int, epsilon: float, Q: np.ndarray, num_actions: int
) -> np.ndarray:
    probs = [epsilon / (num_actions)] * num_actions
    greedy_action_idx = np.argmax(Q[state])
    probs[greedy_action_idx] += 1.0 - epsilon
    return np.array(probs)


def expected_sarsa(
    env: gym.Env,
    gamma: float,
    alpha: float,
    epsilon: float,
    num_episodes: int,
) -> np.ndarray:
    # Notice - since we don't have access to P, we sample from env instead
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    # Arbitrarily initialize Q and pi
    Q = np.zeros((num_states, num_actions))
    policy = {s: np.random.choice(num_actions) for s in range(num_states)}

    for _ in range(num_episodes):
        state = env.reset()
        # Sample a step with e-greedy
        done = False
        while not done:
            # Sample a step and select next action
            action = (
                policy[state]
                if np.random.random() > epsilon
                else np.random.randint(num_actions)
            )
            next_state, reward, done, _ = env.step(action)

            # Get action probabilities for next_state
            next_action_probs = get_action_probs(next_state, epsilon, Q, num_actions)
            
            # Calculate the target
            expected_next_q = np.sum(next_action_probs * Q[next_state])
            target = reward + gamma * expected_next_q * (not done)
            
            # Updated estimated Q
            Q[state][action] = Q[state][action] + alpha * (target - Q[state][action])

            # extract policy - policy imporvement w.r.t to visited states
            # this is the only difference between prediction and control
            policy[state] = np.argmax(Q[state])

            state = next_state
        
        # Decay parameters
        alpha = exponential_decay(alpha, decay_factor=0.99975, min_val=0.001)
        epsilon = exponential_decay(epsilon, decay_factor=0.99975, min_val=0.001)
    return Q, policy


def double_q_learning(
    env: gym.Env,
    gamma: float,
    alpha: float,
    epsilon: float,
    num_episodes: int,
) -> np.ndarray:
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    # Arbitrarily initialize Q and pi
    Q = np.zeros((num_states, num_actions))
    Q1 = np.zeros((num_states, num_actions))
    Q2 = np.zeros((num_states, num_actions))
    policy = {s: np.random.choice(num_actions) for s in range(num_states)}
    
    for _ in range(num_episodes):
        state = env.reset()
        done = False
        while not done:
            # Sample a step with e-greedy in Q1 and Q2
            action = (
                policy[state]
                if np.random.random() > epsilon
                else np.random.randint(num_actions)
            )
            next_state, reward, done, _ = env.step(action)

            # Calculate the target and update either Q1 or Q2
            if np.random.random() < 0.5:
                target = reward + gamma * Q2[next_state][np.argmax(Q1[next_state])] * (
                    not done
                )
                Q1[state][action] = Q1[state][action] + alpha * (
                    target - Q1[state][action]
                )
            else:
                target = reward + gamma * Q1[next_state][np.argmax(Q2[next_state])] * (
                    not done
                )
                Q2[state][action] = Q2[state][action] + alpha * (
                    target - Q2[state][action]
                )

            # extract policy - policy imporvement w.r.t to visited states
            # this is the only difference between prediction and control
            # Update Q, which is the average of Q1 and Q2
            Q[state][action] = np.mean([Q1[state][action], Q2[state][action]])
            policy[state] = np.argmax(Q[state])

            state = next_state

        # Decay parameters
        alpha = exponential_decay(alpha, decay_factor=0.99975, min_val=0.001)
        epsilon = exponential_decay(epsilon, decay_factor=0.99975, min_val=0.001)
    return Q, policy


def double_sarsa(
    env: gym.Env,
    gamma: float,
    alpha: float,
    epsilon: float,
    num_episodes: int,
) -> np.ndarray:
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    # Arbitrarily initialize Q and pi
    Q = np.zeros((num_states, num_actions))
    Q1 = np.zeros((num_states, num_actions))
    Q2 = np.zeros((num_states, num_actions))
    policy = {s: np.random.choice(num_actions) for s in range(num_states)}

    for _ in range(num_episodes):
        state = env.reset()
        # Sample a step with e-greedy in Q1 and Q2
        action = (
            policy[state]
            if np.random.random() > epsilon
            else np.random.randint(num_actions)
        )
        done = False
        while not done:
            next_state, reward, done, _ = env.step(action)

            # Sample the next action with e-greedy in Q1 and Q2
            next_action = (
                policy[next_state]
                if np.random.random() > epsilon
                else np.random.randint(num_actions)
            )

            # Calculate the target and update either Q1 or Q2
            if np.random.random() < 0.5:
                target = reward + gamma * Q2[next_state][next_action] * (not done)
                Q1[state][action] = Q1[state][action] + alpha * (
                    target - Q1[state][action]
                )
            else:
                target = reward + gamma * Q1[next_state][next_action] * (not done)
                Q2[state][action] = Q2[state][action] + alpha * (
                    target - Q2[state][action]
                )

            # extract policy - policy imporvement w.r.t to visited states
            # this is the only difference between prediction and control
            # Update Q, which is the average
            Q[state][action] = np.mean([Q1[state][action], Q2[state][action]])
            policy[state] = np.argmax(Q[state])

            state = next_state

        # Decay parameters
        epsilon = exponential_decay(epsilon, decay_factor=0.99975, min_val=0.001)
        alpha = exponential_decay(alpha, decay_factor=0.99975, min_val=0.001)
    return Q, policy


if __name__ == "__main__":
    env = gym.make("SlipperyWalkFive-v0")

    policy = {s: 1 for s in range(env.observation_space.n)}

    Q, policy = q_learning(
        env=env, gamma=1.0, alpha=0.5, epsilon=0.2, num_episodes=3000
    )
    V_rounded = [round(x, 2) for x in np.max(Q, axis=1)]

    V_true = [0.0, 0.67, 0.89, 0.96, 0.99, 1.0, 0.0]
    policy_true = {0: 0, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 0}

    if V_rounded == V_true:
        print("Converged to the correct value function!")
    else:
        print("Did not converge to the correct value function!")
        print("Expected: ", V_true)
        print("Actual:   ", V_rounded)
    if policy == policy_true:
        print("Converged to the correct policy!")
    else:
        print("Did not converge to the correct policy!")
        print("Expected: ", policy_true)
        print("Actual:   ", policy)
