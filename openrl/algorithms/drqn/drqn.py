"""
DRQN that can either uses the vanilla DQN network.
We could easily extend this to dueling-DQN, DDQN, or dueling-DDQN if we wanted.

"""

import gym
import time
import argparse
import numpy as np
import tensorflow as tf
from typing import Callable, Tuple
from algorithms.drqn.models import drqn_discrete_network
from algorithms.drqn.utils import ReplayBuffer
from util.plotting import plot_training_results


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

        self.history_length = model_kwargs.get("history_length", 4)
        # Actor and target actor models
        self.model = model_fn(state_dims=self.state_dims,
                              num_actions=self.num_actions,
                              num_timesteps=self.history_length,
                              num_hidden_fc_layers=num_hidden_layers,
                              hidden_size=hidden_size)
        self.target_model = model_fn(state_dims=self.state_dims,
                                     num_actions=self.num_actions,
                                     num_timesteps=self.history_length,
                                     num_hidden_fc_layers=num_hidden_layers,
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
            # Expand the dimensions so its in a batch
            state = np.reshape(state, [1, self.history_length, self.state_dims])

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

    @staticmethod
    def update_states(states: np.ndarray, next_state: np.ndarray) -> np.ndarray:
        """
        For RNNs we keep track of previous states and use the history of states as the input into the model.
        When we interact with th e environment and observe a new state, we must update our states.

        :param states: the current states to update - shape (num_timesteps, state_dim)
        :param next_state: the state to add to `states` - shape (state_dim)
        :return: the update state with `next_state` at the end (i.e. the -1 index) - shape (num_timesteps, state_dim)

        For example:
            state_dim = 4
            num_timesteps = 3

            states = np.array([[1,  2,  3,  4],  <-- oldest state
                               [5,  6,  7,  8],
                               [9, 10, 11, 12]]  <-- most recent state

            next_state = np.array([13, 14, 15, 16])

            updated_state = np.array([[5,   6,  7,  8],
                                      [9,  10, 11, 12],
                                      [13, 14, 15, 16]]
        """
        states = np.roll(states, -1, axis=0)  # Shift everything up to make room for the next state
        states[-1] = next_state  # The newest state is always at the end!
        return states

    def train_episode(self) -> dict:
        ep_rewards = 0
        init_state = np.zeros([self.history_length, self.state_dims])
        state = self.update_states(init_state, self.env.reset())
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

            # Update states with next_state
            next_state = self.update_states(state, next_state)

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

        logs = dict()
        logs['ep_rewards'] = ep_rewards
        logs['ep_steps'] = cur_step
        logs['ep_total_loss'] = critic_loss if batch_transitions else None
        return logs

    def run_agent(self, render=False) -> Tuple[float, int]:
        total_reward, total_steps = 0, 0
        init_state = np.zeros([self.history_length, self.state_dims])
        state = self.update_states(init_state, self.env.reset())
        done = False

        while not done:
            if render:
                self.env.render()

            # Select action
            action = self.get_action(state)

            # Interact with environment
            next_state, reward, done, _ = self.env.step(action)

            # Update states with next_state
            next_state = tf.expand_dims(next_state, axis=0)
            state = self.update_states(state, next_state)

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
    buffer = ReplayBuffer(state_dims=_state_dims, action_dims=_action_dims, history_length=args.history_length)

    # Select network architecture
    model_func = drqn_discrete_network

    # Instantiate optimizer
    opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    # Create agent
    agent = DQNAgent(environment=env,
                     model_fn=model_func,
                     optimizer=opt,
                     replay_buffer=buffer,
                     model_kwargs=dict(history_length=args.history_length,
                                       state_dims=_state_dims,
                                       num_actions=_num_actions,
                                       num_hidden_layers=2,
                                       hidden_size=256),
                     train_kwargs=dict(target_update_freq=20,
                                       use_polyak=False),
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
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--history_length", type=int, default=4)
    parser.add_argument("--model_checkpoint_dir", type=str, default="./model_chkpt")
    args = parser.parse_args()

    main()
