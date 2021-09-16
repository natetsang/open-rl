"""
VPG using two NN models.
"""
import gym
import time
import argparse
import numpy as np
import tensorflow as tf
from typing import Union, Callable, Tuple
from models.models import actor_fc_discrete_network, critic_fc_network
from algorithms.vpg.utils import plot_training_results
from util.compute_returns import compute_returns_simple

# Set up constants
GAMMA = 0.99
ACTOR_LEARNING_RATE = 0.001
CRITIC_LEARNING_RATE = 0.001


class VPGAgent:
    def __init__(self,
                 environment: gym.Env,
                 actor_model_fn: Callable[..., tf.keras.Model],
                 actor_optimizer: tf.keras.optimizers,
                 critic_model_fn: Callable[..., tf.keras.Model],
                 critic_optimizer: tf.keras.optimizers,
                 model_kwargs: dict = None,
                 train_kwargs: dict = None,
                 save_dir: str = None) -> None:
        # Env vars
        self.env = environment
        self.state_dims = model_kwargs.get('state_dims')
        self.num_actions = model_kwargs.get('num_actions')

        # Model vars
        self.actor_model = actor_model_fn(state_dims=self.state_dims,
                                          num_actions=self.num_actions,
                                          num_hidden_layers=model_kwargs.get("num_hidden_layers"),
                                          hidden_size=model_kwargs.get("hidden_size"))
        self.critic_model = critic_model_fn(state_dims=self.state_dims,
                                            num_hidden_layers=model_kwargs.get("num_hidden_layers"),
                                            hidden_size=model_kwargs.get("hidden_size"))

        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer

        # Training vars
        self.batch_size = train_kwargs.get("batch_size", 1)
        self.save_dir_actor = save_dir + "_actor"
        self.save_dir_critic = save_dir + "_critic"

    def save_models(self) -> None:
        # TODO >> Fix - Getting error
        # self.actor_model.save(self.save_dir_actor)
        # self.critic_model.save(self.save_dir_critic)
        pass

    def load_models(self) -> Tuple[tf.keras.Model, tf.keras.Model]:
        self.actor_model = tf.keras.models.load_model(self.save_dir_actor)
        self.critic_model = tf.keras.models.load_model(self.save_dir_critic)
        return self.actor_model, self.critic_model

    def train_episode(self) -> Tuple[Union[float, int], int]:
        with tf.GradientTape() as actor_tape, tf.GradientTape() as critic_tape:
            ep_rewards = [0 for _ in range(self.batch_size)]
            ep_actor_loss, ep_critic_loss = [], []
            cur_step = 0
            for i in range(self.batch_size):
                state = self.env.reset()
                done = False
                reward_trajectory, state_trajectory, action_prob_trajectory = [], [], []
                value_trajectory = []
                # Rollout policy to get a single trajectory
                while not done:
                    cur_step += 1
                    # Get state in correct format
                    state = tf.expand_dims(tf.convert_to_tensor(state), 0)
                    state_trajectory.append(state)

                    # Predict action prob and take action
                    action_prob = self.actor_model(state)
                    action = np.random.choice(self.num_actions, p=np.squeeze(action_prob))
                    values = self.critic_model(state)

                    state, reward, done, _ = self.env.step(action)

                    # Some bookkeeping
                    ep_rewards[i] += reward
                    value_trajectory.append(values)
                    reward_trajectory.append(tf.cast(tf.reshape(reward, (1, 1)), tf.float32))
                    action_prob_trajectory.append(tf.convert_to_tensor([tf.expand_dims(action_prob[0][action], 0)]))

                # Calculate rewards
                returns = compute_returns_simple(rewards=reward_trajectory, gamma=GAMMA)

                # Concat
                returns = tf.concat(returns, axis=0)
                action_prob_trajectory = tf.concat(action_prob_trajectory, axis=0)
                value_trajectory = tf.concat(value_trajectory, axis=0)

                # Calculate advantages
                advantages = returns - value_trajectory

                # Calculate losses
                actor_loss = tf.reduce_sum(-tf.math.log(action_prob_trajectory) * tf.stop_gradient(advantages))
                critic_loss = tf.reduce_sum(tf.square(advantages))
                ep_actor_loss.append(actor_loss)
                ep_critic_loss.append(critic_loss)

            # Find mean loss from batch
            mean_actor_trajectory_loss = tf.reduce_mean(ep_actor_loss)
            mean_critic_trajectory_loss = tf.reduce_mean(ep_critic_loss)

        # Backpropagate loss
        actor_grads = actor_tape.gradient(mean_actor_trajectory_loss, self.actor_model.trainable_variables)
        self.actor_optimizer.apply_gradients(zip(actor_grads, self.actor_model.trainable_variables))

        critic_grads = critic_tape.gradient(mean_critic_trajectory_loss, self.critic_model.trainable_variables)
        self.critic_optimizer.apply_gradients(zip(critic_grads, self.critic_model.trainable_variables))

        return float(np.mean(ep_rewards)), cur_step

    def run_agent(self, render=False) -> Tuple[float, int]:
        total_reward, total_steps = 0, 0
        state = self.env.reset()
        done = False

        while not done:
            if render:
                self.env.render()

            # Select action
            action_prob = self.actor_model(tf.expand_dims(state, axis=0))
            action = np.argmax(np.squeeze(action_prob))

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
    _num_actions = env.action_space.n

    # Create agent
    actor_opt = tf.keras.optimizers.Adam(learning_rate=ACTOR_LEARNING_RATE)
    critic_opt = tf.keras.optimizers.Adam(learning_rate=CRITIC_LEARNING_RATE)

    agent = VPGAgent(environment=env,
                     actor_model_fn=actor_fc_discrete_network,
                     actor_optimizer=actor_opt,
                     critic_model_fn=critic_fc_network,
                     critic_optimizer=critic_opt,
                     model_kwargs=dict(state_dims=_state_dims,
                                       num_hidden_layers=2,
                                       hidden_size=128,
                                       num_actions=_num_actions),
                     train_kwargs=dict(batch_size=args.batch_size),
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

        if running_reward > 195:
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
    parser.add_argument("--env", type=str, default="CartPole-v0")
    parser.add_argument("--epochs", type=int, default=250)
    parser.add_argument("--batch_size", type=int, default=5)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--model_checkpoint_dir", type=str, default="./model_chkpt")
    args = parser.parse_args()

    main()
