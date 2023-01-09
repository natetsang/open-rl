import gym
import gym_walk  # noqa
import numpy as np


def exponential_decay(param: float, decay_factor: float, min_val: float) -> float:
    if param > min_val:
        param *= decay_factor
        param = max(param, min_val)
    return param


def computed_discounted_rewards(rewards: list[float], gamma: float = 0.95) -> float:
    """
    Compute the rewards-to-go, which are the cumulative rewards from t=t' to T.
    """
    discounted_reward = 0
    for i, r in enumerate(rewards):
        discounted_reward += gamma**i * r
    return discounted_reward


def nstep_prediction(
    policy: dict[int, int],
    env: gym.Env,
    gamma: float,
    alpha: float,
    n_steps: int,
    num_episodes: int,
) -> np.ndarray:
    num_states = env.observation_space.n

    # Arbitrarily initialize V
    V = np.zeros(num_states)

    for _ in range(num_episodes):
        state = env.reset()
        done = False
        trajectory = []
        while True:
            trajectory = trajectory[1:]  # deque first element

            # Gather and append n-steps (or until done)
            while len(trajectory) < n_steps and not done:
                action = policy[state]
                next_state, reward, done, _ = env.step(action)
                trajectory.append((state, action, reward, done))
                state = next_state

            # TODO(ntsang): this is slightly different than GDRL
            if len(trajectory) == 0:
                break

            # get the first state in the n-step traj, which is what we want to update
            state_to_update = trajectory[0][0]  # (s, a, r, d) tuple

            # get all rewards, r(t=t'), r(t=t'+1), ... r(t=t'+n-1)
            n_rewards = np.array(trajectory)[:, 2]

            # Compute the target by summing discounted n-step rewards and bootstrapping
            n_rewards = computed_discounted_rewards(n_rewards, gamma)
            target = n_rewards + gamma ** (len(trajectory) - 1) * V[next_state] * (
                not done
            )

            # Update estimated V for state_to_update, NOT state
            V[state_to_update] = V[state_to_update] + alpha * (
                target - V[state_to_update]
            )

        # Decay parameters
        alpha = exponential_decay(alpha, decay_factor=0.99975, min_val=0.001)
    return V


def nstep_sarsa(
    env: gym.Env,
    gamma: float,
    alpha: float,
    epsilon: float,
    n_steps: int,
    num_episodes: int,
) -> np.ndarray:
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    # Arbitrarily initialize V and pi
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
        trajectory = []
        while True:
            trajectory = trajectory[1:]  # deque first element

            # Gather and append n-steps (or until done)
            while len(trajectory) < n_steps and not done:
                next_state, reward, done, _ = env.step(action)
                next_action = (
                    policy[next_state]
                    if np.random.random() > epsilon
                    else np.random.randint(num_actions)
                )
                trajectory.append((state, action, reward, next_state, done))
                state, action = next_state, next_action

            if len(trajectory) == 0:
                break

            # get the first state-action in the n-step traj, which is what we want to update
            state_to_update = trajectory[0][0]  # (s, a, r, d) tuple
            action_to_update = trajectory[0][1]

            # get all rewards, r(t=t'), r(t=t'+1), ... r(t=t'+n-1)
            n_rewards = np.array(trajectory)[:, 2]
            n_rewards = computed_discounted_rewards(n_rewards, gamma)
            target = n_rewards + gamma ** (len(trajectory) - 1) * Q[next_state][
                next_action
            ] * (not done)

            # Update estimated Q for state_to_update, NOT state
            Q[state_to_update][action_to_update] = Q[state_to_update][
                action_to_update
            ] + alpha * (target - Q[state_to_update][action_to_update])

            # extract policy - policy imporvement w.r.t to visited states
            # this is the only difference between prediction and control
            policy[state_to_update] = np.argmax(Q[state_to_update])

        # Decay parameters
        alpha = exponential_decay(alpha, decay_factor=0.99975, min_val=0.001)
        epsilon = exponential_decay(epsilon, decay_factor=0.99975, min_val=0.001)
        # print(f"alpha={alpha} --- epsilon={epsilon}")
    return Q, policy


def get_action_probs(
    state: int, epsilon: float, Q: np.ndarray, num_actions: int
) -> np.ndarray:
    probs = [epsilon / (num_actions)] * num_actions
    greedy_action_idx = np.argmax(Q[state])
    probs[greedy_action_idx] += 1.0 - epsilon
    return np.array(probs)


# TODO(ntsang): Maybe add an implementation non-tree backup n-step Q-learning?
# TODO(ntsang): Maybe try importance sampling algorithm (section 7.5)

def nstep_q_learning_tree_backup(
    env: gym.Env,
    gamma: float,
    alpha: float,
    epsilon: float,
    n_steps: int,
    num_episodes: int,
) -> np.ndarray:
    num_states = env.observation_space.n
    num_actions = env.action_space.n

    # Arbitrarily initialize V and pi
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
        trajectory = []
        while True:
            trajectory = trajectory[1:]  # deque first element

            # Gather and append n-steps (or until done)
            while len(trajectory) < n_steps and not done:
                next_state, reward, done, _ = env.step(action)
                next_action = (
                    policy[state]
                    if np.random.random() > epsilon
                    else np.random.randint(num_actions)
                )
                trajectory.append(
                    (state, action, reward, next_state, next_action, done)
                )
                state, action = next_state, next_action

            if len(trajectory) == 0:
                break

            # get the first state-action in the n-step traj, which is what we want to update
            state_to_update = trajectory[0][0]  # (s, a, r, s', a', d) tuple
            action_to_update = trajectory[0][1]

            for k in range(len(trajectory) - 1, -1, -1):
                (
                    k_state,
                    k_action,
                    k_reward,
                    k_next_state,
                    k_next_action,
                    k_done,
                ) = trajectory[k]

                # Get action probs for next_state s'
                next_action_probs = get_action_probs(
                    k_next_state, epsilon, Q, num_actions
                )
                # Get prob for next_action a' actually taken
                next_action_prob_taken = next_action_probs[k_next_action]
                # Get probs for all next_actions a' that weren't taken
                next_action_probs_not_taken = np.delete(
                    next_action_probs, k_next_action
                )
                # Get Q(next_state) for next_actions that weren't actually taken
                Q_next_state_for_actions_not_taken = np.delete(
                    Q[k_next_state], k_next_action
                )

                if k == len(trajectory) - 1:
                    # n=1 -- (case 7.15 pg 153)
                    # this is identical to expected SARSA, for the last timestep
                    G = reward + gamma * np.sum(next_action_probs * Q[next_state]) * (
                        not done
                    )
                else:
                    # n>=2 -- (case 7.16 pg 153)
                    first_term = k_reward + gamma * np.sum(
                        next_action_probs_not_taken * Q_next_state_for_actions_not_taken
                    )
                    second_term = gamma * next_action_prob_taken * G  # recursive
                    G = first_term + second_term

            # This remains unchanged from before
            Q[state_to_update][action_to_update] = Q[state_to_update][
                action_to_update
            ] + alpha * (G - Q[state_to_update][action_to_update])

            # extract policy - policy imporvement w.r.t to visited states
            # this is the only difference between prediction and control
            policy[state_to_update] = np.argmax(Q[state_to_update])

        # Decay parameters
        alpha = exponential_decay(alpha, decay_factor=0.99975, min_val=0.001)
        epsilon = exponential_decay(epsilon, decay_factor=0.99975, min_val=0.001)
    return Q, policy


if __name__ == "__main__":
    env = gym.make("SlipperyWalkFive-v0")

    Q, policy = nstep_q_learning_tree_backup(
        env=env, gamma=1.0, alpha=0.5, epsilon=0.2, n_steps=3, num_episodes=3000
    )

    # Round to make it easier to read
    V_actual = [round(x, 2) for x in np.max(Q, axis=1)]
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
