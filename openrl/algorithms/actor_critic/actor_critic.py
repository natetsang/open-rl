"""
N-step actor-critic using shared NN model with GAE and entropy.
"""
import gym
import time
import argparse
import numpy as np
import tensorflow as tf
from typing import Union, Callable, Tuple
from models.models import actor_critic_fc_discrete_network
from algorithms.actor_critic.utils import plot_training_results
from util.compute_returns import compute_discounted_returns, compute_gae_returns


# Set up constants
GAMMA = 0.99
LAMBDA = 0.95
LEARNING_RATE = 0.001
ACTOR_LOSS_WEIGHT = 1.0
CRITIC_LOSS_WEIGHT = 0.5
ENTROPY_LOSS_WEIGHT = 0.01


class ActorCriticAgent:
    def __init__(self,
                 environment: gym.Env,
                 model_fn: Callable[..., tf.keras.Model],
                 optimizer: tf.keras.optimizers,
                 model_kwargs: dict = None,
                 train_kwargs: dict = None,
                 save_dir: str = None) -> None:
        # Env vars
        self.env = environment
        self.state_dims = model_kwargs.get('state_dims')
        self.num_actions = model_kwargs.get('num_actions')

        # Model vars
        self.model = model_fn(state_dims=self.state_dims,
                              num_actions=self.num_actions,
                              num_hidden_layers=model_kwargs.get("num_hidden_layers"),
                              hidden_size=model_kwargs.get("hidden_size"))
        self.optimizer = optimizer

        # Training vars
        self.max_steps_per_epoch = train_kwargs.get("max_steps_per_epoch", 1000)
        self.n_steps = train_kwargs.get("n_steps", 10)
        self.use_gae = train_kwargs.get("use_gae", True)

        self.save_dir = save_dir

    def save_model(self) -> None:
        self.model.save(self.save_dir)

    def load_model(self) -> tf.keras.Model:
        self.model = tf.keras.models.load_model(self.save_dir)
        return self.model

    def train_episode(self) -> Tuple[Union[float, int], int]:
        ep_rewards = 0
        state = self.env.reset()
        done = False
        total_steps = 0
        while not done:
            cur_step = 0
            reward_trajectory, state_trajectory, mask_trajectory = [], [], []
            action_trajectory, prob_trajectory, action_prob_trajectory = [], [], []
            value_trajectory = []
            with tf.GradientTape() as tape:
                while cur_step < self.n_steps and not done:
                    cur_step += 1
                    total_steps += 1
                    # Get state in correct format
                    state = tf.expand_dims(tf.convert_to_tensor(state), 0)
                    state_trajectory.append(state)

                    # Predict action prob and take action
                    action_prob, values = self.model(state)
                    action = np.random.choice(self.num_actions, p=np.squeeze(action_prob))

                    state, reward, done, _ = self.env.step(action)

                    # Some bookkeeping
                    ep_rewards += reward
                    action_trajectory.append(action)
                    value_trajectory.append(values)
                    reward_trajectory.append(tf.cast(tf.reshape(reward, (1, 1)), tf.float32))
                    mask_trajectory.append(tf.cast(tf.reshape(1 - done, (1, 1)), tf.float32))
                    prob_trajectory.append(action_prob)
                    action_prob_trajectory.append(tf.convert_to_tensor([tf.expand_dims(action_prob[0][action], 0)]))

                _, next_value = self.model(tf.expand_dims(tf.convert_to_tensor(state), 0))
                returns = compute_gae_returns(next_value=next_value, rewards=reward_trajectory, masks=mask_trajectory,
                                              values=value_trajectory, gamma=GAMMA, lambda_=LAMBDA)
                targets = compute_discounted_returns(next_value=next_value, rewards=reward_trajectory,
                                                     masks=mask_trajectory, gamma=GAMMA)

                # Concat
                returns = tf.concat(returns, axis=0)
                targets = tf.concat(targets, axis=0)

                prob_trajectory = tf.concat(prob_trajectory, axis=0)
                action_prob_trajectory = tf.concat(action_prob_trajectory, axis=0)
                value_trajectory = tf.concat(value_trajectory, axis=0)
                advantages = returns - value_trajectory

                # Calculate losses
                actor_loss = -tf.math.log(action_prob_trajectory) * tf.stop_gradient(advantages)
                # There are different values we could use. We could use the GAE advantage or
                # instead we could just do the n-step targets - value_trajectory
                critic_loss = tf.square(advantages)
                entropy_loss = tf.reduce_sum(prob_trajectory * tf.math.log(prob_trajectory + 1e-8), axis=1)
                total_loss = tf.reduce_sum(actor_loss * ACTOR_LOSS_WEIGHT +
                                           critic_loss * CRITIC_LOSS_WEIGHT +
                                           entropy_loss * ENTROPY_LOSS_WEIGHT)

            # Backpropagate loss
            grads = tape.gradient(total_loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return ep_rewards, total_steps

    def run_agent(self, render=False) -> Tuple[float, int]:
        total_reward, total_steps = 0, 0
        state = self.env.reset()
        done = False

        while not done:
            if render:
                self.env.render()

            # Select action
            action_prob, _ = self.model(tf.expand_dims(state, axis=0))
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
    opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    agent = ActorCriticAgent(environment=env,
                             model_fn=actor_critic_fc_discrete_network,
                             optimizer=opt,
                             model_kwargs=dict(state_dims=_state_dims,
                                               num_hidden_layers=2,
                                               hidden_size=128,
                                               num_actions=_num_actions),
                             train_kwargs=dict(max_steps_per_epoch=1000,
                                               use_gae=args.use_gae,
                                               n_steps=args.n_steps),
                             save_dir=args.model_checkpoint_dir,
                             )

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
            agent.save_model()

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
    parser.add_argument("--epochs", type=int, default=600)
    parser.add_argument("--use_gae", type=bool, default=True),
    parser.add_argument("--n_steps", type=int, default=5),
    parser.add_argument("--seed", type=int)
    parser.add_argument("--model_checkpoint_dir", type=str, default="./model_chkpt")
    args = parser.parse_args()

    main()
