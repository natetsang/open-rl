import gym
import gym_walk  # noqa
import numpy as np


def generate_episode(
    policy: dict[int, int], env: gym.Env, epsilon: float = 0.0
) -> list[tuple[int, int, float, bool]]:
    done = False
    num_actions = env.action_space.n
    trajectory = []
    state = env.reset()

    while not done:
        action = (
            policy[state]
            if np.random.random() > epsilon
            else np.random.randint(num_actions)
        )
        state, reward, done, _ = env.step(action)
        trajectory.append((state, action, reward, done))
    return trajectory


def computed_discounted_rewards(rewards: list, gamma: float = 0.95) -> float:
    """
    Compute the rewards-to-go, which are the cumulative rewards from t=t' to T.
    """
    discounted_reward = 0
    for i, r in enumerate(rewards):
        discounted_reward += gamma**i * r
    return discounted_reward


def compute_G_forward_view_lambda(
    rewards: list, V, states, gamma: float = 0.95, lambda_: float = 0.9
) -> float:
    num_steps_to_go = len(rewards)
    G_lambda = 0

    # First term
    for n in range(1, num_steps_to_go):
        state_n = states[n]
        lambda_n = lambda_ ** (n - 1)
        rewards_n = rewards[:n]
        G_n = (
            (1 - lambda_n)
            * computed_discounted_rewards(rewards=rewards_n, gamma=gamma)
            * gamma**n
            * V[state_n]
        )
        G_lambda += G_n

    # Second term
    lambda_T = lambda_ ** (num_steps_to_go - 1)
    G_T = computed_discounted_rewards(rewards=rewards, gamma=gamma)
    G_lambda += lambda_T * G_T

    return G_lambda


def exponential_decay(param: float, decay_factor: float, min_val: float) -> float:
    if param > min_val:
        param *= decay_factor
        param = max(param, min_val)
    return param


def forward_view_td_lambda(
    policy: dict[int, int],
    env: gym.Env,
    gamma: float,
    alpha: float,
    lambda_: float,
    num_episodes: int,
) -> np.ndarray:
    num_states = env.observation_space.n

    # Arbitrarily initialize V
    V = np.zeros(num_states)

    for _ in range(num_episodes):
        # For each episode, reinitialize G
        G = 0

        # Sample an episode trajectory
        episode = generate_episode(policy, env)

        # Loop through each step in the episode
        for t in range(len(episode)):
            states = episode[t:][0]  # (s, a, r, d) tuple
            state = states[0]
            rewards = episode[t:, 2]

            # Compute the target with forward-view TD(lambda)
            G = compute_G_forward_view_lambda(rewards, V, states, gamma, lambda_)

            # Update estimated V
            V[state] = V[state] + alpha * (G - V[state])

        # Decay parameters
        alpha = exponential_decay(alpha, decay_factor=0.99975, min_val=0.001)
    return V


def backward_view_td_lambda(
    policy: dict[int, int],
    env: gym.Env,
    gamma: float,
    alpha: float,
    lambda_: float,
    num_episodes: int,
    replacing_trace: bool,
) -> np.ndarray:
    num_states = env.observation_space.n

    # Arbitrarily initialize V
    V = np.zeros(num_states)

    for _ in range(num_episodes):
        E = np.zeros(num_states)
        state = env.reset()
        done = False
        while not done:
            # sample a step
            action = policy[state]
            next_state, reward, done, _ = env.step(action)

            # Calculate target
            target = reward + gamma * V[next_state] * (not done)

            # Update trace
            E[state] = E[state] + 1
            if replacing_trace:
                E = E.clip(0, 1)

            # Update the entire V function, not just V[state]
            V = V + alpha * (target - V[state]) * E

            # Decay eligibility trace
            E = gamma * lambda_ * E

        # Decay parameters
        alpha = exponential_decay(alpha, decay_factor=0.99975, min_val=0.001)
        state = next_state

    return V


def sarsa_lambda(
    env: gym.Env,
    gamma: float,
    alpha: float,
    lambda_: float,
    epsilon: float,
    num_episodes: int,
    replacing_trace: bool = True,
) -> np.ndarray:
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    # Arbitrarily initialize Q, pi
    Q = np.zeros((num_states, num_actions))
    policy = {s: np.random.choice(num_actions) for s in range(num_states)}

    for _ in range(num_episodes):
        E = np.zeros((num_states, num_actions))
        state = env.reset()
        action = (
            policy[state]
            if np.random.random() > epsilon
            else np.random.randint(num_actions)
        )
        done = False
        while not done:
            # sample a step
            next_state, reward, done, _ = env.step(action)
            next_action = (
                policy[next_state]
                if np.random.random() > epsilon
                else np.random.randint(num_actions)
            )

            # Calculate target
            target = reward + gamma * Q[next_state][next_action] * (not done)

            # Update trace
            E[state][action] = E[state][action] + 1
            if replacing_trace:
                E = E.clip(0, 1)

            # We update the entire Q function, not just Q[state][action]
            Q = Q + alpha * (target - Q[state][action]) * E

            # Decay eligibility trace
            E = gamma * lambda_ * E

            # extract policy - policy imporvement w.r.t to visited states
            # this is the only difference between prediction and control
            # notice that we update the entire policy not just policy[state]
            policy = {s: np.argmax(Q[state]) for s in range(num_states)}

            state, action = next_state, next_action

        # Decay parameters
        alpha = exponential_decay(alpha, decay_factor=0.99975, min_val=0.001)
        epsilon = exponential_decay(epsilon, decay_factor=0.99975, min_val=0.001)
    return Q, policy


def watkins_q_learning(
    env: gym.Env,
    gamma: float,
    alpha: float,
    epsilon: float,
    lambda_: float,
    num_episodes: int,
    replacing_trace: bool = True,
) -> np.ndarray:
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    
    # Arbitrarily initialize Q, pi
    Q = np.zeros((num_states, num_actions))
    policy = {s: np.random.choice(num_actions) for s in range(num_states)}

    for _ in range(num_episodes):
        E = np.zeros((num_states, num_actions))
        state = env.reset()
        action = (
            policy[state]
            if np.random.random() > epsilon
            else np.random.randint(num_actions)
        )
        done = False
        while not done:
            next_state, reward, done, _ = env.step(action)
            next_action = (
                policy[next_state]
                if np.random.random() > epsilon
                else np.random.randint(num_actions)
            )
            next_action_is_greedy = Q[next_state][next_action] == Q[next_state].max()

            # Calculate target
            target = reward + gamma * np.max(Q[next_state]) * (not done)
            
            # Update trace
            E[state][action] = E[state][action] + 1
            if replacing_trace:
                E = E.clip(0, 1)
           
            # We update the entire Q function, not just Q[state][action]
            Q = Q + alpha * (target - Q[state][action]) * E

            # Decay trace: If not greedy, cut traces to zero
            if next_action_is_greedy:
                E *= gamma * lambda_
            else:
                E.fill(0)

            # extract policy - policy imporvement w.r.t to visited states
            # this is the only difference between prediction and control
            # notice that we update the entire policy not just policy[state]
            policy = {s: np.argmax(Q[state]) for s in range(num_states)}

            state, action = next_state, next_action

        # Decay parameters
        alpha = exponential_decay(alpha, decay_factor=0.99975, min_val=0.001)
        epsilon = exponential_decay(epsilon, decay_factor=0.99975, min_val=0.001)
    return Q, policy


if __name__ == "__main__":
    env = gym.make("SlipperyWalkFive-v0")

    Q, policy = watkins_q_learning(
        env=env,
        gamma=1.0,
        alpha=0.5,
        lambda_=0.5,
        epsilon=0.2,
        num_episodes=3000,
        replacing_trace=True,
    )

    # Round to make it easier to read
    V_actual = [round(x, 2) for x in np.max(Q, axis=1)]
    # first and last states are terminal, so prune these
    V_actual = V_actual[1:5]
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