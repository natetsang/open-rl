"""
Continuous VPG using two networks.
This is implemented correctly but doesn't learn. We need a better algo
for continuous action spaces!
"""
from typing import Callable, Tuple
from models.models import actor_fc_continuous_network, critic_fc_network
from util.plotting import plot_training_results
from util.compute_returns import compute_returns_simple

import gym
import time
import argparse
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions


# Set up constants
GAMMA = 0.99
LEARNING_RATE = 0.01


class VPGAgent:
    def __init__(self,
                 environment: gym.Env,
                 actor_model_fn: Callable[..., tf.keras.Model],
                 critic_model_fn: Callable[..., tf.keras.Model],
                 actor_optimizer: tf.keras.optimizers,
                 critic_optimizer: tf.keras.optimizers,
                 model_kwargs: dict = None,
                 save_dir: str = None) -> None:
        # Env vars
        self.env = environment
        self.state_dims = model_kwargs.get('state_dims')
        self.action_dims = model_kwargs.get('action_dims')

        # Model vars
        self.actor_model = actor_model_fn(state_dims=self.state_dims,
                                          action_dims=self.action_dims,
                                          num_hidden_layers=model_kwargs.get("num_hidden_layers"),
                                          hidden_size=model_kwargs.get("hidden_size"))
        self.actor_optimizer = actor_optimizer

        self.critic_model = critic_model_fn(state_dims=self.state_dims,
                                            num_hidden_layers=model_kwargs.get("num_hidden_layers"),
                                            hidden_size=model_kwargs.get("hidden_size"))
        self.critic_optimizer = critic_optimizer

        # Save directories
        self.save_dir_actor = save_dir + "_actor"
        self.save_dir_critic = save_dir + "_critic"

    def save_models(self) -> None:
        self.actor_model.save(self.save_dir_actor)
        self.critic_model.save(self.save_dir_critic)

    def load_models(self) -> Tuple[tf.keras.Model, tf.keras.Model]:
        self.actor_model = tf.keras.models.load_model(self.save_dir_actor)
        self.critic_model = tf.keras.models.load_model(self.save_dir_critic)
        return self.actor_model, self.critic_model

    def train_episode(self) -> dict:
        ep_rewards = 0
        state = self.env.reset()
        done = False
        cur_step = 0
        reward_trajectory, state_trajectory, action_logprob_trajectory = [], [], []
        value_trajectory = []
        with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
            # Rollout policy to get a single trajectory
            while not done:
                cur_step += 1
                # Get state in correct format
                state = tf.expand_dims(tf.convert_to_tensor(state), 0)
                state_trajectory.append(state)

                # Predict action prob and take action
                mus, stds = self.actor_model(state)
                values = self.critic_model(state)

                dist = tfd.MultivariateNormalDiag(loc=mus, scale_diag=stds)
                action = tf.clip_by_value(dist.sample(), clip_value_min=-2, clip_value_max=2.)
                log_prob = dist.log_prob(action)

                state, reward, done, _ = self.env.step(action[0])

                # Some bookkeeping
                ep_rewards += reward
                value_trajectory.append(values)
                reward_trajectory.append(tf.cast(tf.reshape(reward, (1, 1)), tf.float32))
                action_logprob_trajectory.append(tf.cast(tf.reshape(log_prob, (1, 1)), tf.float32))

            # Calculate rewards
            returns = compute_returns_simple(rewards=reward_trajectory, gamma=GAMMA)

            # Concat
            returns = tf.concat(returns, axis=0)
            action_logprob_trajectory = tf.concat(action_logprob_trajectory, axis=0)
            value_trajectory = tf.concat(value_trajectory, axis=0)

            # Calculate advantages
            advantages = returns - value_trajectory

            # Calculate losses
            actor_loss = -tf.reduce_mean(action_logprob_trajectory * tf.stop_gradient(advantages))
            critic_loss = tf.reduce_mean(tf.square(advantages))

        # Backpropagate loss
        actor_grads = actor_tape.gradient(actor_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor_model.trainable_variables))

        critic_grads = critic_tape.gradient(critic_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic_model.trainable_variables))

        logs = dict()
        logs['ep_rewards'] = ep_rewards
        logs['ep_steps'] = cur_step
        logs['ep_total_loss'] = actor_loss
        return logs

    def run_agent(self, render=False) -> Tuple[float, int]:
        total_reward, total_steps = 0, 0
        state = self.env.reset()
        done = False

        while not done:
            if render:
                self.env.render()

            # Select action
            mu, std = self.actor_model(tf.expand_dims(state, axis=0))
            dist = tfd.MultivariateNormalDiag(loc=mu, scale_diag=std)
            action = tf.clip_by_value(dist.mean(), clip_value_min=-2, clip_value_max=2.)

            # Interact with environment
            state, reward, done, _ = self.env.step(action[0])

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
    _state_dims = env.observation_space.shape[0]
    _action_dims = env.action_space.shape[0]

    # Create agent
    actor_opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    critic_opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    agent = VPGAgent(environment=env,
                     actor_model_fn=actor_fc_continuous_network,
                     critic_model_fn=critic_fc_network,
                     actor_optimizer=actor_opt,
                     critic_optimizer=critic_opt,
                     model_kwargs=dict(state_dims=_state_dims,
                                       action_dims=_action_dims,
                                       num_hidden_layers=2,
                                       hidden_size=128),
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
    parser.add_argument("--env", type=str, default="Pendulum-v0")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--model_checkpoint_dir", type=str, default="./model_chkpt")
    args = parser.parse_args()

    main()
