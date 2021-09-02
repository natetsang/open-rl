"""
Deep Dyna-Q that can either use the vanilla DQN network or a dueling DQN network
"""

import gym
import time
import argparse
import numpy as np
import tensorflow as tf
from typing import Union, Callable, Tuple
from algorithms.dyna_q.models import (dqn_fc_discrete_network, dueling_dqn_fc_discrete_network,
                                      fc_transition_network, fc_reward_network)
from algorithms.dyna_q.utils import plot_training_results
from util.utils import ReplayBuffer


# Set up
GAMMA = 0.99
LEARNING_RATE = 0.0001

# Exploration settings
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001


class DDynaQAgent:
    def __init__(self,
                 environment: gym.Env,
                 q_model_fn: Callable[..., tf.keras.Model],
                 transition_model_fn: Callable[..., tf.keras.Model],
                 reward_model_fn: Callable[..., tf.keras.Model],
                 q_optimizer: tf.keras.optimizers,
                 transition_optimizer: tf.keras.optimizers,
                 reward_optimizer: tf.keras.optimizers,
                 replay_buffer_real: ReplayBuffer,
                 replay_buffer_sim: ReplayBuffer,
                 model_kwargs: dict = None,
                 train_kwargs: dict = None,
                 save_dir: str = None) -> None:
        # Env vars
        self.env = environment
        self.state_dims = model_kwargs.get('state_dims')
        self.action_dims = model_kwargs.get('action_dims')
        self.num_actions = model_kwargs.get('num_actions')

        num_hidden_layers = model_kwargs.get("num_hidden_layers")
        hidden_size = model_kwargs.get("hidden_size")

        # Q-network and target Q-network models
        self.q_model = q_model_fn(state_dims=self.state_dims,
                                  num_actions=self.num_actions,
                                  num_hidden_layers=num_hidden_layers,
                                  hidden_size=hidden_size)
        self.target_q_model = q_model_fn(state_dims=self.state_dims,
                                         num_actions=self.num_actions,
                                         num_hidden_layers=num_hidden_layers,
                                         hidden_size=hidden_size)
        self.q_optimizer = q_optimizer

        # Dynamics model
        self.transition_model = transition_model_fn(state_dims=self.state_dims,
                                                    action_dims=self.action_dims,
                                                    num_hidden_layers=num_hidden_layers,
                                                    hidden_size=hidden_size)
        self.transition_optimizer = transition_optimizer

        # Reward model
        self.reward_model = reward_model_fn(state_dims=self.state_dims,
                                            action_dims=self.action_dims,
                                            num_hidden_layers=num_hidden_layers,
                                            hidden_size=hidden_size)
        self.reward_optimizer = reward_optimizer

        # Replay buffers
        self.replay_buffer_real = replay_buffer_real
        self.replay_buffer_sim = replay_buffer_sim

        # Training vars
        self.cur_episode = 0
        self.total_steps = 0
        self.target_update_freq = train_kwargs.get("target_update_freq", 2)
        self.num_planning_steps_per_iter = train_kwargs.get("num_planning_steps_per_iter", 5)
        self.planning_batch_size = train_kwargs.get("planning_batch_size", 0)
        self.epsilon = 1.0

        if train_kwargs.get("use_polyak"):
            self.use_polyak = train_kwargs.get("use_polyak")
        else:
            self.use_polyak = False

        # Other vars
        self.tau = 0.005

        # Save directories
        self.save_dir = save_dir

    def save_models(self) -> None:
        self.q_model.save(self.save_dir)
        self.target_q_model.save(self.save_dir + "_target")

    def load_models(self) -> Tuple[tf.keras.Model, tf.keras.Model]:
        self.q_model = tf.keras.models.load_model(self.save_dir)
        self.target_q_model = tf.keras.models.load_model(self.save_dir + "_target")
        return self.q_model, self.target_q_model

    def update_target_networks(self, use_polyak: bool = False) -> None:
        """
        Update the weights of the target actor and target critic networks.
        We either take a "slow" approach such that the updates occur according to:
            theta_tgt = tau * theta + (1 - tau) * theta_tgt

        Or we take a "hard" approach and copy them
        """
        if use_polyak:
            new_weights = []
            weights = self.q_model.get_weights()
            target_weights = self.target_q_model.get_weights()

            for weights, target_weights in zip(weights, target_weights):
                new_weights.append(self.tau * weights + (1 - self.tau) * target_weights)
            self.target_q_model.set_weights(new_weights)
        else:
            self.target_q_model.set_weights(self.q_model.get_weights())

    def get_action(self, state, decay=True) -> np.ndarray:
        """
        Based on a given state, return an action using epsilon greedy. If decay is True, then
        subsequently decay epsilon
        :param state: the state for which we want to take an action
        :param decay: boolean indicating whether or not to decay epsilon after running
        :return: action to take
        """
        # Take action given current state
        if np.random.random() > self.epsilon:
            q_values = self.q_model(state)
            action = np.argmax(q_values)  # Take greedy action that maximizes Q
        else:
            action = np.random.randint(0, self.num_actions)  # Take random action

        if decay:
            # Decay epsilon
            self.decay_epsilon()
        return action

    def decay_epsilon(self) -> None:
        if self.epsilon > MIN_EPSILON:
            self.epsilon *= EPSILON_DECAY
            self.epsilon = max(MIN_EPSILON, self.epsilon)

    def direct_rl(self) -> Tuple[Union[float, int], int, np.ndarray]:
        """
        Run "direct RL" which is equivalent to one episode of DQN.
        :return:
        """
        ep_rewards = 0
        state = tf.expand_dims(tf.convert_to_tensor(self.env.reset()), 0)
        done = False
        cur_step = 0

        # Rollout policy to get a single trajectory
        while not done:
            cur_step += 1
            self.cur_episode += 1
            self.total_steps += 1

            # Get action and take step
            action = self.get_action(state)
            next_state, reward, done, _ = self.env.step(action)
            next_state = tf.reshape(next_state, [1, self.state_dims])

            # Some bookkeeping
            ep_rewards += reward

            # Add transition to buffer
            self.replay_buffer_real.store_transition((state, action, reward, next_state, done))

            # Retrieve batch of transitions
            batch_transitions = self.replay_buffer_real.sample()

            # If we don't have batch_size experiences in our buffer, keep collecting samples
            if batch_transitions is None:
                state = next_state
                continue

            states, actions, rewards, next_states, dones = batch_transitions

            # Find Q_targets_max and plug into target function
            q_targets_next_state = self.target_q_model(next_states)
            q_targets_max = tf.reduce_max(q_targets_next_state, axis=1, keepdims=True)
            targets = rewards + GAMMA * (1 - dones) * q_targets_max

            # Update our Q-function using real data
            with tf.GradientTape() as tape:
                # Find the Q-values of our actions
                actions = tf.convert_to_tensor(actions, dtype=tf.int32)
                q_values = self.q_model(states)
                q_values = tf.gather(q_values, actions, batch_dims=1)

                # Calculate the loss!
                critic_loss = tf.reduce_mean(tf.square(targets - q_values))
            grads = tape.gradient(critic_loss, self.q_model.trainable_variables)
            self.q_optimizer.apply_gradients(zip(grads, self.q_model.trainable_variables))

            # Note: The traditional Dyna-Q tabular algo has these two steps here instead of outside
            # self.world_model_learning()
            # self.planning(initial_state=state)

            # Delay policy by only updating it every d steps
            if self.use_polyak or self.total_steps % self.target_update_freq == 0:
                self.update_target_networks(use_polyak=self.use_polyak)

            state = next_state
        return ep_rewards, cur_step, state

    def world_model_learning(self) -> None:
        """
        Retrieve a batch of experiences and run supervised learning to train a dynamics/transition
        and a reward model. If we don't have enough samples in the buffer yet, do nothing.
        :return:
        """
        # Retrieve batch of transitions
        batch_transitions = self.replay_buffer_real.sample()

        # If we don't have batch_size experiences in our buffer, do nothing
        if batch_transitions is None:
            return

        states, actions, rewards, next_states, dones = batch_transitions

        # Update dynamics model
        with tf.GradientTape() as tape:
            pred_next_states = self.transition_model([states, actions])
            loss = tf.reduce_mean(tf.square(next_states - pred_next_states))
        grads = tape.gradient(loss, self.transition_model.trainable_variables)
        self.transition_optimizer.apply_gradients(zip(grads, self.transition_model.trainable_variables))

        # Update rewards model
        with tf.GradientTape() as tape:
            pred_rewards = self.reward_model([states, actions])
            loss = tf.reduce_mean(tf.square(rewards - pred_rewards))
        grads = tape.gradient(loss, self.reward_model.trainable_variables)
        self.reward_optimizer.apply_gradients(zip(grads, self.reward_model.trainable_variables))

    def simulate_trajectory(self, initial_state=None) -> None:
        """
        The goal of this is to generate cheap data without interacting with the environment!
        We then use this data to improve the Q-function.

        Using our dynamics and reward models, simulate a trajectory of "self.planning_batch_size" steps.
        If an initial_state is given, start the trajectory from that state. Else sample a state from our
        replay buffer.
        This is called trajectory sampling!
        Note: empirically I've had a lot of trouble getting this to work well.
        Note: we don't consider whether a state we reach is `done`. This was a decision I made because
            I found it hard to train the model to predict `done`. This is also consistent with gdrl implementation.

        :param initial_state: the state we want to start the trajectory in
        :return:
        """
        if initial_state is None:
            # Sample state and action from replay buffer
            transitions_real = self.replay_buffer_real.sample(1)
            if transitions_real is None:
                return
            state, _, _, _, done = transitions_real
            if done:
                self.simulate_trajectory()
        else:
            state = initial_state
            done = False

        for i in range(self.planning_batch_size):
            self.total_steps += 1

            # Get action and take step
            action = self.get_action(state, decay=False)
            next_state = self.transition_model([state, tf.convert_to_tensor([action])])
            next_state = tf.reshape(next_state, [1, self.state_dims])

            reward = self.reward_model([state, tf.convert_to_tensor([action])])

            self.replay_buffer_sim.store_transition((state, action, reward, next_state, done))

            # Delay policy by only updating it every d steps
            if self.use_polyak or self.total_steps % self.target_update_freq == 0:
                self.update_target_networks(use_polyak=self.use_polyak)

            state = next_state

    def simulate_random_states(self) -> None:
        """
        The goal of this is to generate cheap data without interacting with the environment!
        We then use this data to improve the Q-function.

        Randomly sample a batch of transitions from real experiences and run them through the
        transition and reward models.
        This is more closely linked to the tabular Dyna-Q algorithm than `simulate_trajectory()`!

        Note: there's also no clear way to update the target function because we take a batch of steps at a time,
            so doing self.total_steps += 1 won't work for this, like is does when we take one step at a time.
        :return:
        """
        # Sample state and action from replay buffer
        transitions_real = self.replay_buffer_real.sample(self.planning_batch_size)

        if transitions_real is None:
            return

        states, _, _, _, dones = transitions_real
        actions = []
        for s in states:
            a = self.get_action(tf.reshape(s, [1, self.state_dims]), decay=False)
            actions.append(a)
        actions = np.array(actions)

        # Use model to simulate next state, reward, and done
        next_states = self.transition_model([states, tf.convert_to_tensor(actions)])
        rewards = self.reward_model([states, tf.convert_to_tensor(actions)])

        # Add transition to simulation buffer
        # TODO >> Could use store_transition_batch()
        for i in range(len(states)):
            self.replay_buffer_sim.store_transition((states[i], actions[i], rewards[i], next_states[i], dones[i]))

        # Not really sure how to handle target updates for the batch case...
        # self.update_target_networks(use_polyak=self.use_polyak)

    def planning(self, use_trajectory_sampling=False, initial_state=None) -> None:
        """
        This method is a major component of Dyna and is ultimately what separates it from Q-learning.
        First, we generate cheap data using our models. There are two strategies we can employ to do this.
        Second, we use this simulated data to improve our Q-function. To do this, we sample a batch of data
        from our replay buffer, then calculate our loss and take a gradient step.

        :param use_trajectory_sampling: boolean indicating whether to generate simulated data using trajectory
            sampling or instead by getting a random batch of states.
        :param initial_state: if using trajectory sampling, we can pass in the initial state
        :return:
        """
        # Step 1: Select simulation strategy and generate cheap data using our models
        if use_trajectory_sampling:
            self.simulate_trajectory(initial_state=initial_state)
        else:
            self.simulate_random_states()

        # Step 2: Improve Q-function using simulated data
        # Randomly sample previously observed state in simulation
        transition_sim = self.replay_buffer_sim.sample(batch_size=self.planning_batch_size)

        # If we don't have batch_size experiences in our buffer, keep collecting samples
        if transition_sim is None:
            return

        s_sim, a_sim, r_sim, ns_sim, _ = transition_sim
        a_sim = tf.convert_to_tensor(a_sim, dtype=tf.int32)

        # Find Q_targets_max and plug into target function
        q_targets_next_state = self.target_q_model(ns_sim)
        q_targets_max = tf.reduce_max(q_targets_next_state, axis=1, keepdims=True)
        targets = r_sim + GAMMA * q_targets_max  # Note: I removed `done` from this

        # Update our Q-function using simulated data
        with tf.GradientTape() as tape:
            # Find the Q-values of our actions
            q_values = self.q_model(s_sim)
            q_values = tf.gather(q_values, a_sim, batch_dims=1)

            # Calculate the loss!
            critic_loss = tf.reduce_mean(tf.square(targets - q_values))
        grads = tape.gradient(critic_loss, self.q_model.trainable_variables)
        self.q_optimizer.apply_gradients(zip(grads, self.q_model.trainable_variables))

    def train_episode(self) -> Tuple[Union[float, int], int]:
        """
        Run 1 episode of Dyna-Q.
        In this implementation, I'm training the models and doing the planning AFTER direct RL.
        In the traditional tabular Dyna-Q, those steps are doing inside the direct RL loop. I tried that
        too and empirically it performed worse from both computation time and rewards.
        :return:
        """
        # Step 1: Direct RL
        trajectory_rewards, trajectory_steps, s = self.direct_rl()

        # Step 2: Improve the dynamics model
        self.world_model_learning()

        # Step 3: Planning
        for _ in range(self.num_planning_steps_per_iter):
            self.planning(use_trajectory_sampling=False, initial_state=s)

        return trajectory_rewards, trajectory_steps


def main() -> None:
    # Create environment
    env = gym.make(args.env)

    # Set seeds
    if args.seed:
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)
        env.seed(args.seed)

    # Create helper vars for model creation
    _state_dims = len(env.observation_space.high)
    _action_dims = 1
    _num_actions = env.action_space.n

    # Create Replay Buffer
    # Note that we could instead use 1 buffer for both >> "buffer_sim = buffer_real"
    buffer_real = ReplayBuffer(state_dims=_state_dims, action_dims=_action_dims)
    buffer_sim = ReplayBuffer(state_dims=_state_dims, action_dims=_action_dims)

    # Select action-value network architecture for policy
    q_model_func = dueling_dqn_fc_discrete_network if args.network_architecture == "dueling" else dqn_fc_discrete_network

    # Select dynamics and reward model network architectures
    # Note that we could instead use 1 model with two heads instead of separate models
    transition_model_func = fc_transition_network
    reward_model_func = fc_reward_network

    # Instantiate optimizers
    q_opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    transition_opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    reward_opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    # Create agent
    agent = DDynaQAgent(environment=env,
                        q_model_fn=q_model_func,
                        transition_model_fn=transition_model_func,
                        reward_model_fn=reward_model_func,
                        q_optimizer=q_opt,
                        transition_optimizer=transition_opt,
                        reward_optimizer=reward_opt,
                        replay_buffer_real=buffer_real,
                        replay_buffer_sim=buffer_sim,
                        model_kwargs=dict(state_dims=_state_dims,
                                          action_dims=_action_dims,
                                          num_actions=_num_actions,
                                          num_hidden_layers=2,
                                          hidden_size=256),
                        train_kwargs=dict(target_update_freq=20,
                                          use_polyak=False,
                                          num_planning_steps_per_iter=args.num_planning_steps_per_iter,
                                          planning_batch_size=args.planning_batch_size),
                        save_dir=args.model_checkpoint_dir)

    # Run training
    best_mean_rewards = -1e8
    running_reward = 0
    ep_rewards_history = []
    ep_steps_history = []
    ep_running_rewards_history = []
    ep_wallclock_history = []
    start = time.time()
    for e in range(args.epochs):
        ep_rew, ep_steps = agent.train_episode()

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
        if latest_mean_rewards > best_mean_rewards:
            best_mean_rewards = latest_mean_rewards
            agent.save_models()

        if running_reward > 495:
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
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--seed", type=int, default=10)
    parser.add_argument("--network_architecture", type=str, default="dqn")
    parser.add_argument("--model_checkpoint_dir", type=str, default="./model_chkpt")
    parser.add_argument("--num_planning_steps_per_iter", type=int, default=1)  # DDyna-Q is equivalent to DQN at K=0
    parser.add_argument("--planning_batch_size", type=int, default=16)  # Batch size for planning q-learning update
    args = parser.parse_args()

    main()
