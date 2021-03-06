"""
DQN that can either use the vanilla DQN network or a dueling DQN network
"""

import gym
import numpy as np
import tensorflow as tf
from typing import Union, Callable, Tuple
from util.replay_buffer import ReplayBuffer


# Set up
GAMMA = 0.99
LEARNING_RATE = 0.0001

# Exploration settings
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001


class DQNAgent:
    def __init__(self,
                 environment: gym.Env,
                 model_fn: Callable[..., tf.keras.Model],
                 optimizer: tf.keras.optimizers,
                 replay_buffer: ReplayBuffer,
                 model_kwargs: dict = None,
                 train_kwargs: dict = None,
                 save_dir: str = None) -> None:
        # Env vars
        self.env = environment
        self.state_dims = model_kwargs.get('state_dims')
        self.num_actions = model_kwargs.get('num_actions')

        num_hidden_layers = model_kwargs.get("num_hidden_layers")
        hidden_size = model_kwargs.get("hidden_size")

        # Actor and target actor models
        self.model = model_fn(state_dims=self.state_dims,
                              num_actions=self.num_actions,
                              num_hidden_layers=num_hidden_layers,
                              hidden_size=hidden_size)
        self.target_model = model_fn(state_dims=self.state_dims,
                                     num_actions=self.num_actions,
                                     num_hidden_layers=num_hidden_layers,
                                     hidden_size=hidden_size)
        self.optimizer = optimizer

        # Replay buffer
        self.replay_buffer = replay_buffer

        # Training vars
        self.cur_episode = 0
        self.total_steps = 0
        self.target_update_freq = train_kwargs.get("target_update_freq", 2)
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
        self.model.save(self.save_dir)
        self.target_model.save(self.save_dir + "_target")

    def load_models(self) -> Tuple[tf.keras.Model, tf.keras.Model]:
        self.model = tf.keras.models.load_model(self.save_dir)
        self.target_model = tf.keras.models.load_model(self.save_dir + "_target")
        return self.model, self.target_model

    def update_target_networks(self, use_polyak: bool = False) -> None:
        """
        Update the weights of the target actor and target critic networks.
        We either take a "slow" approach such that the updates occur according to:
            theta_tgt = tau * theta + (1 - tau) * theta_tgt

        Or we take a "hard" approach and copy them
        """
        if use_polyak:
            new_weights = []
            weights = self.model.get_weights()
            target_weights = self.target_model.get_weights()

            for weights, target_weights in zip(weights, target_weights):
                new_weights.append(self.tau * weights + (1 - self.tau) * target_weights)
            self.target_model.set_weights(new_weights)
        else:
            self.target_model.set_weights(self.model.get_weights())

    def get_action(self, state):
        # Take action given current state
        if np.random.random() > self.epsilon:
            q_values = self.model(state)
            action = np.argmax(q_values)  # Take greedy action that maximizes Q
        else:
            action = np.random.randint(0, self.num_actions)  # Take random action

        # Decay epsilon
        self.decay_epsilon()
        return action

    def decay_epsilon(self) -> None:
        if self.epsilon > MIN_EPSILON:
            self.epsilon *= EPSILON_DECAY
            self.epsilon = max(MIN_EPSILON, self.epsilon)

    def train_episode(self) -> Tuple[Union[float, int], int]:
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
            self.replay_buffer.store_transition((state, action, reward, next_state, done))

            # Retrieve batch of transitions
            batch_transitions = self.replay_buffer.sample()

            # If we don't have batch_size experiences in our buffer, keep collecting samples
            if batch_transitions is None:
                state = next_state
                continue

            states, actions, rewards, next_states, dones = batch_transitions

            # Find Q_targets_max and plug into target function
            q_targets_next_state = self.target_model(next_states)
            q_targets_max = tf.reduce_max(q_targets_next_state, axis=1, keepdims=True)
            targets = rewards + GAMMA * (1 - dones) * q_targets_max

            # Update our Q-function
            with tf.GradientTape() as tape:
                # Find the Q-values of our actions
                actions = tf.convert_to_tensor(actions, dtype=tf.int32)
                q_values = self.model(states)
                q_values = tf.gather(q_values, actions, batch_dims=1)

                # Calculate the loss!
                critic_loss = tf.reduce_mean(tf.square(targets - q_values))
            grads = tape.gradient(critic_loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            # Delay policy by only updating it every d steps
            if self.use_polyak or self.total_steps % self.target_update_freq == 0:
                self.update_target_networks(use_polyak=self.use_polyak)

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
            q_a_values = self.model(tf.expand_dims(state, axis=0))
            action = np.argmax(np.squeeze(q_a_values))

            # Interact with environment
            state, reward, done, _ = self.env.step(action)

            # Bookkeeping
            total_reward += reward
            total_steps += 1
        return total_reward, total_steps
