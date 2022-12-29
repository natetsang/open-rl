import gym
import numpy as np


def exponential_decay(param: float, decay_factor: float, min_val: float) -> float:
    if param > min_val:
        param *= decay_factor
        param = max(decay_factor, min_val)
    return param


def discount_rewards(rewards: list, gamma: float = 0.95) -> float:
    """
    Compute the rewards-to-go, which are the cumulative rewards from t=t' to T.
    """
    discounted_rewards = []
    total_ret = 0
    for r in rewards[::-1]:
        # Without discount
        # total_ret = r + total_ret

        # With discount
        total_ret = r + gamma * total_ret
        discounted_rewards.insert(0, total_ret)
    return np.sum(discounted_rewards)


def nstep_prediction(
    policy: dict[int, int],
    env: gym.Env,
    gamma: float,
    alpha: float,
    n_steps: int,
    num_episodes: int,
) -> np.ndarray:
    num_states = env.observation_space.n
    V = np.zeros(num_states)

    for _ in range(num_episodes):
        state = env.reset()
        done = False
        trajectory = []
        while True:
            trajectory = trajectory[1:]  # remove first element

            # Gather and append steps
            while len(trajectory) < n_steps and not done:
                action = policy[state]
                next_state, reward, done, _ = env.step(action)
                trajectory.append((state, action, reward, done))
                state = next_state

            # TODO(ntsang): this is slightly different than GDRL
            # I think the 'and done' is unnecessary here
            if len(trajectory) == 0 and done:
                break

            state_to_update = trajectory[0][0]  # get the first state in the n-step traj
            n_rewards = np.array(trajectory)[:, 2]  # get all rewards, r(t=0), r(t=1), ... r(t=n-1)
            n_discounted_rewards: float = discount_rewards(n_rewards, gamma)
            target = n_discounted_rewards + gamma**(len(n_discounted_rewards) - 1) * V[next_state] * (not done)
            V[state_to_update] = V[state_to_update] + alpha * (target - V[state_to_update])
        alpha = exponential_decay(alpha, decay_factor=0.99975, min_val=0.001)  # seems out of place, perhaps remove?
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
    Q = np.zeros((num_states, num_actions))
    policy = {}
    for _ in range(num_episodes):
        state = env.reset()
        action = (
            policy[state]
            if np.random.random() > epsilon
            else np.random.randint(num_actions)
        )
        epsilon = exponential_decay(epsilon, decay_factor=0.99975, min_val=0.001)
        done = False
        trajectory = []
        while True:
            trajectory = trajectory[1:]  # remove first element

            # Gather and append steps
            while len(trajectory) < n_steps and not done:
                next_state, reward, done, _ = env.step(action)
                next_action = (
                    policy[next_state]
                    if np.random.random() > epsilon
                    else np.random.randint(num_actions)
                )
                trajectory.append((state, action, reward, done))
                state, action = next_state, next_action

            if len(trajectory) == 0 and done:
                break

            state_to_update = trajectory[0][0]  # get the first state in the n-step traj
            n_rewards = np.array(trajectory)[:, 2]  # get all rewards, r(t=0), r(t=1), ... r(t=n-1)
            n_discounted_rewards: float = discount_rewards(n_rewards, gamma)
            target = n_discounted_rewards + gamma**(len(n_discounted_rewards) - 1) * Q[next_state][next_action] * (not done)
            Q[state_to_update][action] = Q[state_to_update][action] + alpha * (target - Q[state_to_update][action])

            # TODO(ntsang): this is slightly different than GDRL
            # I think the 'and done' is unnecessary here

            # extract policy - policy imporvement w.r.t to visited states
            # this is the only difference between prediction and control
            policy[state] = np.argmax(Q[state])

        alpha = exponential_decay(alpha, decay_factor=0.99975, min_val=0.001)  # seems out of place, perhaps remove?
    return Q, policy


def get_action_probs(state, epsilon, Q, num_actions):
    probs = [
        epsilon / (num_actions)
    ] * num_actions
    greedy_action_idx = np.argmax(Q[state])
    probs[greedy_action_idx] += (
        1.0 - epsilon
    )
    return np.array(probs)


# TODO(ntsang): This is incorrect -- needs to be n-step tree backup algorithm
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
    Q = np.zeros((num_states, num_actions))
    policy = {}
    for _ in range(num_episodes):
        state = env.reset()
        action = (
            policy[state]
            if np.random.random() > epsilon
            else np.random.randint(num_actions)
        )
        epsilon = exponential_decay(epsilon, decay_factor=0.99975, min_val=0.001)
        done = False
        trajectory = []
        while True:
            trajectory = trajectory[1:]  # remove first element

            # Gather and append steps
            while len(trajectory) < n_steps and not done:
                next_state, reward, done, _ = env.step(action)
                next_action = (
                    policy[state]
                    if np.random.random() > epsilon
                    else np.random.randint(num_actions)
                )
                trajectory.append((state, action, reward, next_state, done))
                state, action = next_state, next_action

            # TODO(ntsang): this is slightly different than GDRL
            # I think the 'and done' is unnecessary here
            if len(trajectory) == 0 and done:
                break

            state_to_update = trajectory[0][0]  # get the first state in the n-step traj

            # TODO(ntsang): something's probably off here with edge cases where len(traj) < n
            G = 0
            for k in range(len(trajectory) - 1, -1, -1):
                k_state, k_action, k_reward, k_next_state, k_done = trajectory[k]
                k_action_probs = get_action_probs(k_state, epsilon, Q, num_actions)
                k_next_action_greedy = np.argmax(Q[k_next_state])
                k_next_action_probs_not_greedy = np.delete(k_action_probs, k_next_action_greedy)
                Q_not_greedy = np.delete(Q[k_next_state], k_next_action_greedy)

                first_term = k_reward + gamma * np.sum(k_next_action_probs_not_greedy * Q_not_greedy)
                second_term = gamma * k_action_probs[k_next_action_greedy] * G
                G = first_term + second_term

            # This remains unchanged from before
            target = G
            Q[state_to_update][action] = Q[state_to_update][action] + alpha * (target - Q[state_to_update][action])

            # extract policy - policy imporvement w.r.t to visited states
            # this is the only difference between prediction and control
            policy[state] = np.argmax(Q[state])

        alpha = exponential_decay(alpha, decay_factor=0.99975, min_val=0.001)  # seems out of place, perhaps remove?
    return Q, policy
