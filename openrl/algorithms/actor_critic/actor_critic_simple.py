"""
1-step actor-critic using shared NN model.
"""
import gym
import time
import argparse
import numpy as np
import tensorflow as tf
from typing import Union, List, Callable, Tuple
from models.models import actor_critic_fc_discrete_network
from .utils import plot_training_results


# Set up constants
GAMMA = 0.99
LEARNING_RATE = 0.001
ACTOR_LOSS_WEIGHT = 1.0
CRITIC_LOSS_WEIGHT = 0.5


def compute_discounted_returns(next_value, rewards: List, masks: List) -> List:
    """
    :param next_value:
    :param rewards:
    :param masks:
    :return:
    """
    discounted_rewards = []
    total_ret = next_value * masks[-1]
    for r in rewards[::-1]:
        total_ret = r + GAMMA * total_ret
        discounted_rewards.insert(0, total_ret)
    return discounted_rewards


class ActorCriticAgent:
    def __init__(self,
                 environment: gym.Env,
                 model_fn: Callable[..., tf.keras.Model],
                 optimizer: tf.keras.optimizers,
                 model_kwargs: dict = None,
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
        while not done:
            cur_step = 0
            reward_trajectory, state_trajectory, mask_trajectory = [], [], []
            action_prob_trajectory = []
            value_trajectory = []
            with tf.GradientTape() as tape:
                # Take 1 step, then boostrap
                cur_step += 1
                # Get state in correct format
                state = tf.expand_dims(tf.convert_to_tensor(state), 0)
                state_trajectory.append(state)

                # Predict action prob and take action
                action_prob, values = self.model(state)
                action = np.random.choice(self.num_actions, p=np.squeeze(action_prob))

                state, reward, done, _ = self.env.step(action)

                # Some bookkeeping
                ep_rewards += reward
                value_trajectory.append(values)
                reward_trajectory.append(tf.cast(tf.reshape(reward, (1, 1)), tf.float32))
                mask_trajectory.append(tf.cast(tf.reshape(1 - done, (1, 1)), tf.float32))
                action_prob_trajectory.append(tf.convert_to_tensor([tf.expand_dims(action_prob[0][action], 0)]))

                _, next_value = self.model(tf.expand_dims(tf.convert_to_tensor(state), 0))
                returns = compute_discounted_returns(next_value, reward_trajectory, mask_trajectory)

                # Concat
                returns = tf.concat(returns, axis=0)
                action_prob_trajectory = tf.concat(action_prob_trajectory, axis=0)
                value_trajectory = tf.concat(value_trajectory, axis=0)
                advantages = returns - value_trajectory

                # Calculate losses
                actor_loss = -tf.math.log(action_prob_trajectory) * tf.stop_gradient(advantages)
                critic_loss = tf.square(advantages)
                total_loss = tf.reduce_mean(actor_loss * ACTOR_LOSS_WEIGHT +
                                            critic_loss * CRITIC_LOSS_WEIGHT)

            # Backpropagate loss
            grads = tape.gradient(total_loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return ep_rewards, cur_step

    def test_agent(self, render=False) -> Union[float, int]:
        total_reward = 0
        state = self.env.reset()
        done = False
        while not done:
            if render:
                self.env.render()
            action_prob, _ = self.model(tf.expand_dims(tf.convert_to_tensor(state), 0))
            action = np.argmax(np.squeeze(action_prob))
            state, reward, done, _ = self.env.step(action)
            total_reward += reward
        return total_reward


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
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--model_checkpoint_dir", type=str, default="./model_chkpt")
    args = parser.parse_args()

    main()
