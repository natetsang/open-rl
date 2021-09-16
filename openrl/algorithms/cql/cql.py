"""
Conservative Q-Learning (CQL)

TODO:
    > Fix plotting
    > Instead of online_epochs, be able to change the number of steps
    > Fix the target weight updates so they work without poly_ak
    >
"""

import gym
import time
import argparse
import numpy as np
import tensorflow as tf
from copy import deepcopy
from typing import Callable, Tuple
from algorithms.cql.dqn import DQNAgent
from algorithms.cql.models import dqn_fc_discrete_network
from util.replay_buffer import ReplayBuffer


# Set up
GAMMA = 0.99
LEARNING_RATE = 0.0001


class CQLAgent:
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
        self.prev_steps = 0
        self.target_update_freq = train_kwargs.get("target_update_freq", 2)
        self.cql_alpha = train_kwargs.get("cql_alpha", 0.0)

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

    def train_episode(self) -> dict:
        self.cur_episode += 1
        info = dict()

        # Retrieve batch of transitions
        batch_transitions = self.replay_buffer.sample()

        # If we don't have batch_size experiences in our buffer, keep collecting samples
        if batch_transitions is None:
            return info

        states, actions, rewards, next_states, dones = batch_transitions

        prev_steps = self.total_steps
        self.total_steps += len(states)

        # Find Q_targets_max and plug into target function
        q_targets_next_state = self.target_model(next_states)
        q_targets_max = tf.reduce_max(q_targets_next_state, axis=1, keepdims=True)
        targets = rewards + GAMMA * (1 - dones) * q_targets_max

        # Update our Q-function
        with tf.GradientTape() as tape:
            # Find the Q-values of our actions
            actions = tf.convert_to_tensor(actions, dtype=tf.int32)

            # Q-value for all (s,a) pairs
            q_a_values = self.model(states)

            # Q-value for action actually taken in dataset
            q_values = tf.gather(q_a_values, actions, batch_dims=1)

            # Calculate typical DQN loss!
            critic_loss = tf.reduce_mean(tf.square(targets - q_values))

            # Now calculate the CQL loss!
            q_logsumexp = tf.reduce_logsumexp(q_a_values, axis=1)
            cql_loss = tf.reduce_mean(q_logsumexp - q_values)

            # Combine the losses
            loss = critic_loss + self.cql_alpha * cql_loss
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        # Delay policy by only updating it every d steps
        # TODO >> Because we're getting a batch of steps, we might never reach this if we don't set
        #   use_polyak = True. Adjust run_episodes so we can set a max steps
        if self.use_polyak or self.total_steps % self.target_update_freq == 0:
            self.update_target_networks(use_polyak=self.use_polyak)

        # Return info
        info['training loss'] = loss
        info['CQL loss'] = cql_loss
        info['data Q-values'] = tf.reduce_mean(q_values)
        info['OOD Q-values'] = tf.reduce_mean(q_logsumexp)
        info['cumulative steps'] = self.total_steps
        return info

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
    offline_env = gym.make(args.env)

    # Set seeds
    if args.seed:
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)
        env.seed(args.seed)
        offline_env.seed(args.seed)

    # Create helper vars for model creation
    _state_dims = len(env.observation_space.high)
    _action_dims = 1
    _num_actions = env.action_space.n

    # Create Replay Buffer
    buffer = ReplayBuffer(state_dims=_state_dims, action_dims=_action_dims)

    # Select network architecture
    model_func = dqn_fc_discrete_network

    # Instantiate optimizers
    online_opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    offline_opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    # Create online agent for generating data
    online_agent = DQNAgent(environment=env,
                            model_fn=model_func,
                            optimizer=online_opt,
                            replay_buffer=buffer,
                            model_kwargs=dict(state_dims=_state_dims,
                                              num_actions=_num_actions,
                                              num_hidden_layers=2,
                                              hidden_size=256),
                            train_kwargs=dict(target_update_freq=20,
                                              use_polyak=False),
                            save_dir=None)

    # Data generation
    for e in range(args.online_epochs):
        _, _ = online_agent.train_episode()

    print("BUFFER SIZE: ", len(buffer))
    # Create offline agent
    offline_agent = CQLAgent(environment=offline_env,
                             model_fn=model_func,
                             optimizer=offline_opt,
                             replay_buffer=deepcopy(buffer),  # Use buffer with prepopulated data
                             model_kwargs=dict(state_dims=_state_dims,
                                               num_actions=_num_actions,
                                               num_hidden_layers=2,
                                               hidden_size=256),
                             train_kwargs=dict(target_update_freq=20,
                                               use_polyak=True,
                                               cql_alpha=args.cql_alpha),
                             save_dir=args.model_checkpoint_dir)

    # Run offline training
    best_mean_loss = 1e8
    running_loss = 0
    ep_loss_history = []
    ep_steps_history = []
    ep_running_loss_history = []
    ep_wallclock_history = []
    start = time.time()
    for e in range(args.offline_epochs):
        log = offline_agent.train_episode()

        training_loss = log['training loss']
        cql_loss = log['CQL loss']
        data_q_values = log['data Q-values']
        ood_q_values = log['OOD Q-values']
        cumulative_training_steps = log['cumulative steps']

        ep_wallclock_history.append(time.time() - start)

        # Track progress
        if e == 0:
            running_loss = training_loss
        else:
            running_loss = 0.05 * training_loss + (1 - 0.05) * running_loss

        ep_loss_history.append(training_loss)
        ep_running_loss_history.append(running_loss)
        ep_steps_history.append(cumulative_training_steps)

        if e % 10 == 0:
            template = "running loss: {:.2f} | episode loss: {:.2f} at episode {}"
            print(template.format(running_loss, training_loss, e))

        latest_mean_loss = np.mean(ep_running_loss_history[-10:])
        if latest_mean_loss < best_mean_loss:
            best_mean_loss = latest_mean_loss
            offline_agent.save_models()

    print(f"Training time elapsed (sec): {round(time.time() - start, 2)}")

    # Now let's evaluate the trained CQL by running it on the simulator
    cql_evaluation_rewards = []
    dqn_evaluation_rewards = []
    for e in range(args.evaluation_epochs):
        cql_reward = offline_agent.run_agent()
        dqn_reward = online_agent.run_agent()

        cql_evaluation_rewards.append(cql_reward)
        dqn_evaluation_rewards.append(dqn_reward)

    print(f"CQL mean evaluation reward alpha={offline_agent.cql_alpha}: {np.mean(cql_evaluation_rewards)}")
    print("DQN mean evaluation reward: ", np.mean(dqn_evaluation_rewards))

    # Plot summary of results
    # plot_training_results(rewards_history=ep_rewards_history,
    #                       running_rewards_history=ep_running_rewards_history,
    #                       steps_history=ep_steps_history,
    #                       wallclock_history=ep_wallclock_history,
    #                       save_dir="./results.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v0")
    parser.add_argument("--online_epochs", type=int, default=90)
    parser.add_argument("--offline_epochs", type=int, default=1000)
    parser.add_argument("--evaluation_epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument('--cql_alpha', type=float, default=0.5)
    parser.add_argument("--model_checkpoint_dir", type=str, default="./model_chkpt")
    args = parser.parse_args()

    main()
