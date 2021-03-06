import gym
import time
import argparse
import numpy as np
import tensorflow as tf
from typing import Callable, Tuple

from agents.base_agent import BaseAgent
from algorithms.ddpg.models import ddpg_actor_fc_continuous_network, ddpg_critic_fc_continuous_network
from algorithms.ddpg.utils import OUActionNoise
from util.plotting import plot_training_results
from util.replay_buffer import ReplayBuffer

# Set up
GAMMA = 0.99
ACTOR_LEARNING_RATE = 0.0001
CRITIC_LEARNING_RATE = 0.001


class DDPGAgent(BaseAgent):
    def __init__(self,
                 environment: gym.Env,
                 actor_model_fn: Callable[..., tf.keras.Model],
                 actor_optimizer: tf.keras.optimizers,
                 critic_model_fn: Callable[..., tf.keras.Model],
                 critic_optimizer: tf.keras.optimizers,
                 replay_buffer: ReplayBuffer,
                 model_kwargs: dict = None,
                 save_dir: str = None) -> None:
        # Env vars
        self.env = environment
        self.state_dims = model_kwargs.get('state_dims')
        self.action_dims = model_kwargs.get('action_dims')
        self.env_action_lb = self.env.action_space.low[0]
        self.env_action_ub = self.env.action_space.high[0]

        # Actor and target actor models
        self.actor_model = actor_model_fn(state_dims=self.state_dims,
                                          action_dims=self.action_dims,
                                          env_action_lb=self.env_action_lb,
                                          env_action_ub=self.env_action_ub)
        self.target_actor_model = actor_model_fn(state_dims=self.state_dims,
                                                 action_dims=self.action_dims,
                                                 env_action_lb=self.env_action_lb,
                                                 env_action_ub=self.env_action_ub)
        self.target_actor_model.set_weights(self.actor_model.get_weights())
        self.actor_optimizer = actor_optimizer

        # Critic and target critic models
        self.critic_model = critic_model_fn(state_dims=self.state_dims,
                                            action_dims=self.action_dims)
        self.target_critic_model = critic_model_fn(state_dims=self.state_dims,
                                                   action_dims=self.action_dims)
        self.target_critic_model.set_weights(self.critic_model.get_weights())
        self.critic_optimizer = critic_optimizer

        # Replay buffer
        self.replay_buffer = replay_buffer

        # Other vars
        self.tau = 0.001

        # Save directories
        self.save_dir_actor = save_dir + "_actor"
        self.save_dir_critic = save_dir + "_critic"

    def save_models(self) -> None:
        self.actor_model.save(self.save_dir_actor)
        self.critic_model.save(self.save_dir_critic)
        self.target_actor_model.save(self.save_dir_actor + "_target")
        self.target_critic_model.save(self.save_dir_critic + "_target")

    def load_models(self) -> Tuple[tf.keras.Model, tf.keras.Model, tf.keras.Model, tf.keras.Model]:
        self.actor_model = tf.keras.models.load_model(self.save_dir_actor)
        self.critic_model = tf.keras.models.load_model(self.save_dir_critic)
        self.target_actor_model = tf.keras.models.load_model(self.save_dir_actor + "_target")
        self.target_critic_model = tf.keras.models.load_model(self.save_dir_critic + "_target")
        return self.actor_model, self.target_actor_model, self.critic_model, self.target_critic_model

    def update_target_networks(self) -> None:
        """
        Update the weights of the target actor and target critic networks.
        We take a "slow" approach such that the updates occur according to:
            theta_tgt = tau * theta + (1 - tau) * theta_tgt
        """
        # Actor update
        new_weights = []
        actor_weights = self.actor_model.get_weights()
        target_actor_weights = self.target_actor_model.get_weights()

        for weights, target_weights in zip(actor_weights, target_actor_weights):
            new_weights.append(self.tau * weights + (1 - self.tau) * target_weights)
        self.target_actor_model.set_weights(new_weights)

        # Critic update
        new_weights = []
        critic_weights = self.critic_model.get_weights()
        target_critic_weights = self.target_critic_model.get_weights()

        for weights, target_weights in zip(critic_weights, target_critic_weights):
            new_weights.append(self.tau * weights + (1 - self.tau) * target_weights)
        self.target_critic_model.set_weights(new_weights)

    def train_episode(self, noise: OUActionNoise) -> dict:
        ep_rewards = 0
        state = tf.expand_dims(tf.convert_to_tensor(self.env.reset()), 0)
        done = False
        cur_step = 0
        noise.reset()
        # Rollout policy to get a single trajectory
        while not done:
            cur_step += 1

            # Get action and clip
            action = self.actor_model(state) + noise()
            action = tf.clip_by_value(action, self.env_action_lb, self.env_action_ub)

            # Take step
            next_state, reward, done, _ = self.env.step(action)
            next_state = tf.reshape(next_state, [1, self.state_dims])

            # Some bookkeeping
            ep_rewards += reward[0]
            self.replay_buffer.store_transition((state, action, reward, next_state, done))

            # Retrieve batch of transitions
            batch_transitions = self.replay_buffer.sample()

            # If we don't have batch_size experiences in our buffer, keep collecting samples
            if batch_transitions is None:
                state = next_state
                continue

            states, actions, rewards, next_states, dones = batch_transitions

            # Calculate critic loss
            with tf.GradientTape() as tape:
                target_next_actions = self.target_actor_model(next_states)
                targets = rewards + GAMMA * (1 - dones) * self.target_critic_model([next_states, target_next_actions])
                q_values = self.critic_model([states, actions])
                critic_loss = tf.reduce_mean(tf.square(targets - q_values))
            grads = tape.gradient(critic_loss, self.critic_model.trainable_variables)
            self.critic_optimizer.apply_gradients(zip(grads, self.critic_model.trainable_variables))

            # Calculate actor loss
            with tf.GradientTape() as tape:
                actor_actions = self.actor_model(states)
                critic_values = self.critic_model([states, actor_actions])
                actor_loss = -tf.reduce_mean(critic_values)
            grads = tape.gradient(actor_loss, self.actor_model.trainable_variables)
            self.actor_optimizer.apply_gradients(zip(grads, self.actor_model.trainable_variables))

            # "slow" update of target weights
            self.update_target_networks()
            state = next_state

        logs = dict()
        logs['ep_rewards'] = ep_rewards
        logs['ep_steps'] = cur_step
        logs['ep_total_loss'] = actor_loss if batch_transitions else None
        return logs

    def run_agent(self, render=False) -> Tuple[float, int]:
        total_reward, total_steps = 0, 0
        state = self.env.reset()
        done = False

        while not done:
            if render:
                self.env.render()

            # Select action
            action = self.actor_model(tf.expand_dims(state, axis=0))
            action = tf.clip_by_value(action, self.env_action_lb, self.env_action_ub)

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

    # Create Replay Buffer
    buffer = ReplayBuffer(state_dims=_state_dims, action_dims=_action_dims)

    # Create Noise object
    std_dev = 0.2
    ou_noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

    actor_opt = tf.keras.optimizers.Adam(learning_rate=ACTOR_LEARNING_RATE)
    critic_opt = tf.keras.optimizers.Adam(learning_rate=CRITIC_LEARNING_RATE)

    # Create agent
    agent = DDPGAgent(environment=env,
                      actor_model_fn=ddpg_actor_fc_continuous_network,
                      actor_optimizer=actor_opt,
                      critic_model_fn=ddpg_critic_fc_continuous_network,
                      critic_optimizer=critic_opt,
                      replay_buffer=buffer,
                      model_kwargs=dict(state_dims=_state_dims,
                                        action_dims=_action_dims),
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
        train_logs = agent.train_episode(ou_noise)  # TODO >> Move the noise object into the DDPGAgent instantiation

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

        if running_reward > -250:
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
