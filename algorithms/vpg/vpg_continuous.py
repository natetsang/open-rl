"""
Continuous VPG using two networks.
This is implemented correctly but doesn't learn. We need a better algo
for continuous action spaces!
"""
from typing import Union, List, Callable, Tuple
from models_continuous import actor_fc_continuous_network, critic_fc_continuous_network
from utils import plot_training_results

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


def compute_returns(rewards: List) -> List:
    """
    Compute the rewards-to-go, which are the cumulative rewards from t=t' to T.

    :param rewards: a list of rewards where the ith entry is the reward received at timestep t=i.
    :return: the rewards-to-go, where the ith entry is the cumulative rewards from timestep t=i to t=T,
        where T is equal to len(rewards).
    """
    discounted_rewards = []
    total_ret = 0
    for r in rewards[::-1]:
        # Without discount
        # total_ret = r + total_ret

        # With discount
        total_ret = r + GAMMA * total_ret
        discounted_rewards.insert(0, total_ret)
    return discounted_rewards


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
        self.num_inputs = model_kwargs.get('num_inputs')
        self.num_actions = model_kwargs.get('num_actions')

        # Model vars
        self.actor_model = actor_model_fn(num_inputs=self.num_inputs,
                                          num_hidden_layers=model_kwargs.get("num_hidden_layers"),
                                          hidden_size=model_kwargs.get("hidden_size"),
                                          num_actions=self.num_actions)
        self.actor_optimizer = actor_optimizer

        self.critic_model = critic_model_fn(num_inputs=self.num_inputs,
                                            num_hidden_layers=model_kwargs.get("num_hidden_layers"),
                                            hidden_size=model_kwargs.get("hidden_size"))
        self.critic_optimizer = critic_optimizer

        self.save_dir = save_dir

    def train_episode(self) -> Tuple[Union[float, int], int]:
        ep_rewards = 0
        state = env.reset()
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

                state, reward, done, _ = env.step(action[0])

                # Some bookkeeping
                ep_rewards += reward
                value_trajectory.append(values)
                reward_trajectory.append(tf.cast(tf.reshape(reward, (1, 1)), tf.float32))
                action_logprob_trajectory.append(tf.cast(tf.reshape(log_prob, (1, 1)), tf.float32))

            # Calculate rewards
            returns = compute_returns(reward_trajectory)

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

        return ep_rewards, cur_step


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Pendulum-v0")
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--model_checkpoint_dir", type=str, default="./model_chkpt")
    args = parser.parse_args()

    # Create environment
    env = gym.make(args.env)

    # Set seeds
    if args.seed:
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)
        env.seed(args.seed)

    # Create helper vars for model creation
    _num_inputs = env.observation_space.shape[0]
    _num_actions = env.action_space.shape[0]

    # Create agent
    actor_opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    critic_opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    agent = VPGAgent(environment=env,
                     actor_model_fn=actor_fc_continuous_network,
                     critic_model_fn=critic_fc_continuous_network,
                     actor_optimizer=actor_opt,
                     critic_optimizer=critic_opt,
                     model_kwargs=dict(num_inputs=_num_inputs,
                                       num_hidden_layers=2,
                                       hidden_size=128,
                                       num_actions=_num_actions),
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
        # Train one episode
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

        print(f"running reward: {round(running_reward, 2)} | episode reward: {round(ep_rew, 2)} at episode {e}")
        latest_mean_rewards = np.mean(ep_rewards_history[-10:])

    # Now that we've completed training, let's plot the results
    print(f"Training time elapsed (sec): {round(time.time() - start, 2)}")

    # Plot summary of results
    plot_training_results(rewards_history=ep_rewards_history,
                          running_rewards_history=ep_running_rewards_history,
                          steps_history=ep_steps_history,
                          wallclock_history=ep_wallclock_history,
                          save_dir="./results.png")
