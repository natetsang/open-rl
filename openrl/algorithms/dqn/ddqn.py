"""
Double DQN (DDQN) that can either use the vanilla DQN network or a dueling DQN network
"""

import gym
import time
import argparse
import numpy as np
import tensorflow as tf
from typing import Callable, Tuple
from algorithms.dqn.models import dqn_fc_discrete_network, dueling_dqn_fc_discrete_network
from algorithms.dqn.utils import plot_training_results
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

        if train_kwargs.get("use_huber_loss"):
            self.use_huber_loss = train_kwargs.get("use_huber_loss")
        else:
            self.use_huber_loss = False

        # Other vars
        self.tau = 0.005

        # Save directories
        self.save_dir = save_dir
        self.save_dir_target = save_dir + "_target"

    def save_models(self) -> None:
        self.model.save(self.save_dir)
        self.target_model.save(self.save_dir_target)

    def load_models(self) -> Tuple[tf.keras.Model, tf.keras.Model]:
        self.model = tf.keras.models.load_model(self.save_dir)
        self.target_model = tf.keras.models.load_model(self.save_dir_target)
        return self.model, self.target_model

    def update_target_networks(self, use_polyak: bool = False) -> None:
        """
        Update the weights of the target actor and target critic networks.
        We either take a "slow" approach such that the updates occur according to Polyak averaging:
            theta_tgt = tau * theta + (1 - tau) * theta_tgt

        Or we take a "hard" approach and copy them.
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

    def train_episode(self) -> dict:
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

            # NOTE: I'm not sure if everything below should be inside the tf.GradientTape() or not...it works both ways

            # Get the argmax next actions using the online Q-function (not the target Q-function!)
            q_next_state = self.model(next_states)
            argmax_next_actions = tf.expand_dims(np.argmax(q_next_state, axis=1), axis=1)

            # Using the target Q-function, find the value of the next state using the argmax action
            # from the online Q-function
            q_targets_next_state = self.target_model(next_states)
            q_targets_next_state = tf.gather(q_targets_next_state, argmax_next_actions, batch_dims=1)

            # Calculate targets
            targets = rewards + GAMMA * (1 - dones) * q_targets_next_state

            # Using the batch, calculate the targets and update our Q-function
            with tf.GradientTape() as tape:
                # Find the Q-values of our actions
                q_values = self.model(states)
                actions = tf.convert_to_tensor(actions, dtype=tf.int32)
                q_values = tf.gather(q_values, actions, batch_dims=1)

                # Calculate the loss!
                if self.use_huber_loss:
                    critic_loss = tf.reduce_mean(tf.keras.losses.huber(targets, q_values, delta=1.))
                else:
                    critic_loss = tf.reduce_mean(tf.square(targets - q_values))
            grads = tape.gradient(critic_loss, self.model.trainable_variables)
            # NOTE - instead of Huber Loss, I could clip gradients: grads = [tf.clip_by_norm(g, 10.0) for g in grads]
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

            # Delay policy by only updating it every d steps
            if self.use_polyak or self.total_steps % self.target_update_freq == 0:
                self.update_target_networks(use_polyak=self.use_polyak)

            state = next_state

        logs = dict()
        logs['ep_rewards'] = ep_rewards
        logs['ep_steps'] = cur_step
        logs['ep_total_loss'] = critic_loss if batch_transitions else None
        return logs

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
    buffer = ReplayBuffer(state_dims=_state_dims, action_dims=_action_dims)

    # Select network architecture
    model_func = dueling_dqn_fc_discrete_network if args.network_architecture == "dueling" else dqn_fc_discrete_network

    # Instantiate optimizer
    opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    # Create agent
    agent = DQNAgent(environment=env,
                     model_fn=model_func,
                     optimizer=opt,
                     replay_buffer=buffer,
                     model_kwargs=dict(state_dims=_state_dims,
                                       num_actions=_num_actions,
                                       num_hidden_layers=2,
                                       hidden_size=256),
                     train_kwargs=dict(target_update_freq=20,
                                       use_polyak=False,
                                       use_huber_loss=False),
                     save_dir=args.model_checkpoint_dir)

    # Run training
    best_mean_rewards = -float('inf')
    running_reward = 0
    ep_rewards_history = []
    ep_running_rewards_history = []
    ep_steps_history = []
    ep_loss_history = []
    ep_wallclock_history = []
    start = time.time()
    for e in range(args.epochs):
        # Train one episode
        train_logs = agent.train_episode()

        # Track progress
        ep_rew = train_logs['ep_rewards']
        ep_steps = train_logs['ep_steps']
        ep_losses = train_logs['ep_total_loss']

        running_reward = ep_rew if e == 0 else 0.05 * ep_rew + (1 - 0.05) * running_reward

        ep_rewards_history.append(ep_rew)
        ep_running_rewards_history.append(running_reward)
        ep_steps_history.append(ep_steps)
        ep_loss_history.append(ep_losses)
        ep_wallclock_history.append(time.time() - start)

        if e % 10 == 0:
            print(f"EPISODE {e} | running reward: {running_reward:.2f} - episode reward: {ep_rew:.2f}")

        latest_mean_rewards = np.mean(ep_rewards_history[-10:])
        if latest_mean_rewards > best_mean_rewards:
            best_mean_rewards = latest_mean_rewards
            agent.save_models()

        if running_reward > 195:
            print("Solved at episode {}!".format(e))
            break

    # Now that we've completed training, let's plot the results
    print(f"Training time elapsed (sec): {round(time.time() - start, 2)}")

    # Plot summary of results
    plot_training_results(rewards_history=ep_rewards_history,
                          running_rewards_history=ep_running_rewards_history,
                          steps_history=ep_steps_history,
                          loss_history=ep_loss_history,
                          wallclock_history=ep_wallclock_history,
                          save_dir="./results.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v0")
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--network_architecture", type=str, default="dqn")
    parser.add_argument("--model_checkpoint_dir", type=str, default="./model_chkpt")
    args = parser.parse_args()

    main()
