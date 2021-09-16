"""
Tabular Dyna-Q
@source - Mostly from https://github.com/mimoralea/gdrl/blob/master/notebooks/chapter_07/chapter-07.ipynb
"""

import gym
import time
import argparse
import numpy as np
from typing import Union, Tuple
from algorithms.dyna_q.utils import plot_training_results


# Set up
GAMMA = 0.99
LEARNING_RATE = 0.05

# Exploration settings
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001


class DynaQAgent:
    def __init__(self,
                 environment: gym.Env,
                 model_kwargs: dict = None,
                 train_kwargs: dict = None) -> None:
        # Env vars
        self.env = environment
        self.state_dims = model_kwargs.get('state_dims')
        self.num_actions = model_kwargs.get('num_actions')

        # Q-network
        self.Q_table = np.zeros((self.state_dims, self.num_actions), dtype=np.float32)
        # Dynamics model
        self.transition_table = np.zeros((self.state_dims, self.num_actions, self.state_dims), dtype=np.int)
        # Reward model
        self.reward_table = np.zeros((self.state_dims, self.num_actions, self.state_dims), dtype=np.float32)

        # Training vars
        self.num_planning_steps_per_iter = train_kwargs.get("num_planning_steps_per_iter", 5)
        self.epsilon = 1.0

    def get_action(self, state: np.ndarray, greedy=False, decay=True) -> np.ndarray:
        """
        Based on a given state, return an action using epsilon greedy or greedy. If decay is True, then
        subsequently decay epsilon
        :param greedy:
        :param state: the state for which we want to take an action
        :param decay: boolean indicating whether or not to decay epsilon after running
        :return: action to take
        """
        # Take action given current state
        if greedy or np.random.random() > self.epsilon:
            q_values = self.Q_table[state]
            action = np.argmax(q_values)  # Take greedy action that maximizes Q
        else:
            action = np.random.randint(0, self.num_actions)  # Take random action

        if decay and not greedy:
            self.decay_epsilon()
        return action

    def decay_epsilon(self) -> None:
        if self.epsilon > MIN_EPSILON:
            self.epsilon *= EPSILON_DECAY
            self.epsilon = max(MIN_EPSILON, self.epsilon)

    def world_model_learning(self, transition: Tuple) -> None:
        """
        Update the transition and reward model given a single experience.
        :param transition: (s,a,r,ns,d) tuple to improve the model
        :return:
        """
        state, action, reward, next_state, done = transition

        # Model Update - Using real experience, improve our model
        self.transition_table[state][action][next_state] += 1
        r_diff = reward - self.reward_table[state][action][next_state]
        self.reward_table[state][action][next_state] += (r_diff / self.transition_table[state][action][next_state])

    def planning(self) -> None:
        """
        This method is a major component of Dyna and is ultimately what separates it from Q-learning.
        Randomly sample a state and action that we've already taken. Then simulate the next_state and reward.
        Finally update the Q-function using this simulated experience.
        :return:
        """
        # If we haven't made any updates to the Q-function, there's not much to plan!
        if self.Q_table.sum() == 0:
            return

        for _ in range(self.num_planning_steps_per_iter):
            # Sample s_hat and a_hat arbitrarily
            # We select a state from a list of states that we've already visited
            visited_states = np.where(np.sum(self.transition_table, axis=(1, 2)) > 0)[0]
            state = np.random.choice(visited_states)

            # We select an action from a list of actions that we've already visited from that state
            actions_taken = np.where(np.sum(self.transition_table[state], axis=1) > 0)[0]
            action = np.random.choice(actions_taken)

            # Pass the (s,a) tuple through model to simulate the next state and reward
            prbs = self.transition_table[state][action] / self.transition_table[state][action].sum()  # for stochastic
            next_state = np.random.choice(np.arange(self.state_dims), size=1, p=prbs)[0]
            reward = self.reward_table[state][action][next_state]  # get average reward for (s,a,s') tuple

            # Update Q-table using simulated data
            td_target = reward + GAMMA * self.Q_table[next_state].max()
            td_error = td_target - self.Q_table[state][action]
            self.Q_table[state][action] += LEARNING_RATE * td_error

    def trajectory_planning(self, initial_state: np.ndarray) -> None:
        """
        This method is a major component of Dyna and is ultimately what separates it from Q-learning.
        Randomly sample a state and action that we've already taken. Then simulate the next_state and reward.
        Finally update the Q-function using this simulated experience.
        :return:
        """
        # If we haven't made any updates to the Q-function, there's not much to plan!
        if self.Q_table.sum() == 0:
            return

        state = initial_state
        for _ in range(self.num_planning_steps_per_iter):
            # We select an action from a list of actions that we've already visited from that state
            action = self.get_action(state)

            # If we haven't done the (s,a) before, let's break out
            if self.transition_table[state][action].sum() == 0:
                break

            # Pass the (s,a) tuple through model to simulate the next state and reward
            prbs = self.transition_table[state][action] / self.transition_table[state][action].sum()  # for stochastic
            next_state = np.random.choice(np.arange(self.state_dims), size=1, p=prbs)[0]
            reward = self.reward_table[state][action][next_state]  # get average reward for (s,a,s') tuple

            # Update Q-table using simulated data
            td_target = reward + GAMMA * self.Q_table[next_state].max()
            td_error = td_target - self.Q_table[state][action]
            self.Q_table[state][action] += LEARNING_RATE * td_error

            state = next_state

    def train_episode(self, use_trajectory_sampling=True) -> Tuple[Union[float, int], int]:
        """ Run 1 episode of Dyna-Q. """
        ep_rewards = 0
        state = self.env.reset()
        done = False
        cur_step = 0

        # Rollout policy to get a single trajectory
        while not done:
            # Step 1: Begin Direct RL (identical to normal Q-learning)
            cur_step += 1

            # Get action and take step
            action = self.get_action(state)
            next_state, reward, done, _ = self.env.step(action)

            # Some bookkeeping
            ep_rewards += reward

            # Update Q
            td_target = reward + GAMMA * self.Q_table[next_state].max() * (1 - done)
            td_error = td_target - self.Q_table[state][action]
            self.Q_table[state][action] += LEARNING_RATE * td_error

            # Step 2: Model Update - Using our real experience, improve our model
            self.world_model_learning(transition=(state, action, reward, next_state, done))

            # Step 3: Planning - Generate simulated experience using our model
            # and use it to update the Q-function!
            if use_trajectory_sampling:
                self.trajectory_planning(initial_state=state)
            else:
                self.planning()

            state = next_state
        return ep_rewards, cur_step

    def run_agent(self, render=False) -> Tuple[float, int]:
        total_reward, total_steps = 0, 0
        state = self.env.reset()
        done = False

        while not done:
            if render:
                self.env.render()

            # Select action
            action = self.get_action(state, greedy=True)

            # Interact with environment
            state, reward, done, _ = self.env.step(action)

            # Bookkeeping
            total_reward += reward
            total_steps += 1
        return total_reward, total_steps


def main() -> None:
    # Create environment
    env = gym.make(args.env)

    # Set seeds
    if args.seed:
        np.random.seed(args.seed)
        env.seed(args.seed)

    # Create helper vars for model creation
    _state_dims = env.observation_space.n
    _num_actions = env.action_space.n

    # Create agent
    agent = DynaQAgent(environment=env,
                       model_kwargs=dict(state_dims=_state_dims, num_actions=_num_actions),
                       train_kwargs=dict(num_planning_steps_per_iter=args.num_planning_steps_per_iter))

    # Run training
    best_mean_rewards = -1e8
    running_reward = 0
    ep_rewards_history = []
    ep_steps_history = []
    ep_running_rewards_history = []
    ep_wallclock_history = []
    start = time.time()
    for e in range(args.epochs):
        ep_rew, ep_steps = agent.train_episode(use_trajectory_sampling=False)

        ep_wallclock_history.append(time.time() - start)

        # Track progress
        if e == 0:
            running_reward = ep_rew
        else:
            running_reward = 0.05 * ep_rew + (1 - 0.05) * running_reward

        ep_rewards_history.append(ep_rew)
        ep_running_rewards_history.append(running_reward)
        ep_steps_history.append(ep_steps)

        if e % 10 == 0:
            template = "running reward: {:.2f} | episode reward: {:.2f} at episode {}"
            print(template.format(running_reward, ep_rew, e))

        latest_mean_rewards = np.mean(ep_rewards_history[-10:])

        if running_reward > 0.78:
            print("Solved at episode {}!".format(e))
            break

        # Now that we've completed training, let's plot the results
    print(f"Training time elapsed (sec): {round(time.time() - start, 2)}")

    # Plot summary of results
    plot_training_results(rewards_history=ep_rewards_history,
                          running_rewards_history=ep_running_rewards_history,
                          steps_history=ep_steps_history,
                          wallclock_history=ep_wallclock_history,
                          save_dir="./results.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="FrozenLake-v1")
    parser.add_argument("--epochs", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_planning_steps_per_iter", type=int, default=50)
    args = parser.parse_args()

    main()
