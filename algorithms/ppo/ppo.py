"""
N-step A2C using shared NN model with GAE and entropy.
"""
import gym
import time
import argparse
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp
from typing import List, Callable, Tuple, Generator
from multiprocessing_env import SubprocVecEnv
from models import actor_fc_continuous_network, critic_fc_network
from utils import plot_training_results
tfd = tfp.distributions


# Set up
GAMMA = 0.99
LAMBDA = 0.95
CLIP_PARAM = 0.2
ACTOR_LEARNING_RATE = 3e-4
CRITIC_LEARNING_RATE = 3e-4
ENTROPY_WEIGHT = 0.001

NUM_STEPS_PER_ROLLOUT = 1008
TARGET_KL = 0.01

# This represents one full pass through the buffer
BATCH_SIZE = 64
k_train_iters = 8  # NUM_STEPS_PER_ROLLOUT // MINIBATCH_SIZE


def normalize(x):
    x -= tf.math.reduce_mean(x)
    x /= (tf.math.reduce_std(x) + 1e-8)
    return x


def make_env(env_name: str) -> Callable[[], gym.Env]:
    """
    Given an environment name, return a function that can be called to create an environment.
    """
    def _thunk() -> gym.Env:
        env = gym.make(env_name)
        return env
    return _thunk


def compute_gae_returns(next_value, rewards: List, masks: List, values: List) -> List:
    """
    Computes the generalized advantage estimation (GAE) of the returns.

    :param next_value:
    :param rewards:
    :param masks:
    :param values:
    :return: GAE of the returns
    """
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + GAMMA * values[step + 1] * masks[step] - values[step]
        gae = delta + GAMMA * LAMBDA * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns


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


def sample_batch(mini_batch_size: int, states: tf.Tensor, actions: tf.Tensor, log_probs: tf.Tensor,
                 returns: tf.Tensor, advantage: tf.Tensor) -> Tuple:
    rand_ids = np.random.randint(0, len(states), mini_batch_size)

    return (
        tf.gather(states, rand_ids), tf.gather(actions, rand_ids), tf.gather(log_probs, rand_ids),
        tf.gather(returns, rand_ids), tf.gather(advantage, rand_ids)
    )


def sample_batch_generator(mini_batch_size: int, states: tf.Tensor, actions: tf.Tensor, log_probs: tf.Tensor,
                           returns: tf.Tensor, advantage: tf.Tensor) -> Generator:
    batch_size = len(states)
    for _ in range(len(states) // mini_batch_size):
        rand_ids = np.random.randint(0, batch_size, mini_batch_size)

        yield (
            tf.gather(states, rand_ids), tf.gather(actions, rand_ids), tf.gather(log_probs, rand_ids),
            tf.gather(returns, rand_ids), tf.gather(advantage, rand_ids)
        )


class PPOAgent:
    def __init__(self,
                 environments: SubprocVecEnv,
                 eval_env: gym.Env,
                 actor_model_fn: Callable[..., tf.keras.Model],
                 actor_optimizer: tf.keras.optimizers,
                 critic_model_fn: Callable[..., tf.keras.Model],
                 critic_optimizer: tf.keras.optimizers,
                 model_kwargs: dict = None,
                 train_kwargs: dict = None,
                 save_dir: str = None) -> None:
        # Env vars
        self.envs = environments
        self.eval_env = eval_env
        self.num_inputs = model_kwargs.get('num_inputs')
        self.num_actions = model_kwargs.get('num_actions')
        self.env_action_lb = self.eval_env.action_space.low[0]
        self.env_action_ub = self.eval_env.action_space.high[0]

        # Model vars
        self.actor_model = actor_model_fn(num_inputs=self.num_inputs,
                                          num_hidden_layers=model_kwargs.get("num_hidden_layers"),
                                          hidden_size=model_kwargs.get("hidden_size"),
                                          num_actions=self.num_actions)
        self.critic_model = critic_model_fn(num_inputs=self.num_inputs,
                                            num_hidden_layers=model_kwargs.get("num_hidden_layers"),
                                            hidden_size=model_kwargs.get("hidden_size"))

        self.actor_optimizer = actor_optimizer
        self.critic_optimizer = critic_optimizer

        # Training vars

        self.save_dir = save_dir

    def rollout_policy(self) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor,
                                      tf.Tensor, tf.Tensor, tf.Tensor, tf.Tensor]:
        states = []
        actions = []
        rewards = []
        masks = []
        values = []
        log_probs = []
        for i in range(NUM_STEPS_PER_ROLLOUT):
            state = self.envs.reset()

            # Pass state through actor and critic
            mu, std = self.actor_model(state)
            dist = tfd.MultivariateNormalDiag(loc=mu, scale_diag=std)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            critic_value = self.critic_model(state)

            # Step
            next_state, reward, done, _ = self.envs.step(action)

            # Bookkeeping
            states.append(tf.cast(state, tf.float32))
            actions.append(tf.cast(action, tf.float32))
            rewards.append(tf.cast(tf.reshape(reward, (args.num_envs, 1)), tf.float32))
            masks.append(tf.cast(tf.reshape(1 - done, (args.num_envs, 1)), tf.float32))
            values.append(critic_value)
            log_probs.append(tf.cast(tf.reshape(log_prob, (args.num_envs, 1)), tf.float32))

            state = next_state

        # Calculate A & G
        next_value = self.critic_model(state)
        returns = compute_gae_returns(next_value, rewards, masks, values)

        states = tf.concat(states, axis=0)
        actions = tf.concat(actions, axis=0)
        rewards = tf.concat(rewards, axis=0)
        masks = tf.concat(masks, axis=0)
        values = tf.stop_gradient(tf.concat(values, axis=0))
        log_probs = tf.stop_gradient(tf.concat(log_probs, axis=0))
        returns = tf.stop_gradient(tf.concat(returns, axis=0))

        advantages = returns - values
        # advantages = normalize(advantages)

        return states, actions, rewards, masks, values, log_probs, returns, advantages

    def compute_loss_pi(self, states: tf.Tensor, actions: tf.Tensor,
                        log_probs_old: tf.Tensor, advs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        # Get pi_distribution
        mus, stds = self.actor_model(states)
        dists = tfd.MultivariateNormalDiag(loc=mus, scale_diag=stds)

        # Get log_prob of action taken
        log_probs_new = dists.log_prob(actions)
        log_probs_new = tf.expand_dims(log_probs_new, axis=1)

        # Compute ratios
        ratios = tf.math.exp(log_probs_new - log_probs_old)

        # Compute actor loss
        surr_1 = ratios * advs
        surr_2 = tf.clip_by_value(ratios, 1. - CLIP_PARAM, 1. + CLIP_PARAM) * advs
        actor_loss = -tf.reduce_mean(tf.math.minimum(surr_1, surr_2))

        # Compute entropy loss
        entropy_loss = -tf.reduce_mean(dists.entropy()) * ENTROPY_WEIGHT

        return actor_loss + entropy_loss, log_probs_new

    def train_episode(self) -> None:
        # Step 1: Rollout policy and calculate Advantages and Returns
        states, actions, rewards, masks, values, log_probs_old, rets, advs = self.rollout_policy()

        num_batches_per_iter = len(states) // BATCH_SIZE
        for i in range(k_train_iters):
            for j in range(num_batches_per_iter):
                # Step 2: Sample a batch of observations from the rollouts
                state, action, log_prob_old, ret, adv = sample_batch(BATCH_SIZE, states, actions,
                                                                     log_probs_old, rets, advs)

                # Step 2: Calculate Actor loss and do gradient step
                with tf.GradientTape() as tape:
                    loss, logps_new = self.compute_loss_pi(state, action, log_prob_old, adv)
                    kl = tf.reduce_mean(log_prob_old - logps_new)

                    if kl > 1.5 * TARGET_KL:
                        # print(f"Break at iter {round(i, 3)}/{k_train_iters} because KL {kl} > {1.5 * TARGET_KL}")
                        break
                grads = tape.gradient(loss, self.actor_model.trainable_variables)
                self.actor_optimizer.apply_gradients(zip(grads, self.actor_model.trainable_variables))

                # Step 4: Calculate Critic Loss and do gradient step
                with tf.GradientTape() as tape:
                    value = self.critic_model(state)
                    critic_loss = tf.reduce_mean(tf.square(value - ret))
                grads = tape.gradient(critic_loss, self.critic_model.trainable_variables)
                self.critic_optimizer.apply_gradients(zip(grads, self.critic_model.trainable_variables))

    def test_agent(self) -> float:
        total_rewards = []
        for i in range(10):
            state = tf.expand_dims(tf.convert_to_tensor(self.eval_env.reset()), 0)
            done = False
            ep_rewards = 0
            while not done:
                state = tf.reshape(state, [1, self.num_inputs])
                mu, std = self.actor_model(state)
                dist = tfd.MultivariateNormalDiag(loc=mu, scale_diag=std)
                action = dist.mean()[0]
                state, reward, done, _ = self.eval_env.step(action)
                ep_rewards += reward
            total_rewards.append(ep_rewards)
        return float(np.mean(total_rewards))


def main() -> None:
    # Make single env for testing model performance
    eval_env = gym.make(args.env)

    # Set seeds
    if args.seed:
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)
        eval_env.seed(args.seed)

    # Create helper vars for model creation
    _num_inputs = eval_env.observation_space.shape[0]
    _num_actions = eval_env.action_space.shape[0]

    # Make multiple vectorized environments for synchronous training
    envs_list = [make_env(args.env) for i in range(args.num_envs)]
    envs = SubprocVecEnv(envs_list)

    # Initialize model
    actor_opt = tf.keras.optimizers.Adam(learning_rate=ACTOR_LEARNING_RATE)
    critic_opt = tf.keras.optimizers.Adam(learning_rate=CRITIC_LEARNING_RATE)

    agent = PPOAgent(environments=envs,
                     eval_env=eval_env,
                     actor_model_fn=actor_fc_continuous_network,
                     actor_optimizer=actor_opt,
                     critic_model_fn=critic_fc_network,
                     critic_optimizer=critic_opt,
                     model_kwargs=dict(num_inputs=_num_inputs,
                                       num_hidden_layers=2,
                                       hidden_size=256,
                                       num_actions=_num_actions),
                     train_kwargs=None,
                     save_dir=args.model_checkpoint_dir)

    # Run training
    running_reward = 0
    for e in range(args.epochs):
        agent.train_episode()

        eval_rews = agent.test_agent()
        if e == 0:
            running_reward = eval_rews
        else:
            running_reward = 0.05 * eval_rews + (1 - 0.05) * running_reward

        template = "running reward: {:.2f} | eval reward: {:.2f} at episode {} & step {}"
        print(template.format(running_reward, eval_rews, e, e * NUM_STEPS_PER_ROLLOUT * args.num_envs))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Pendulum-v0")
    parser.add_argument("--num_envs", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--model_checkpoint_dir", type=str, default="./model_chkpt")
    args = parser.parse_args()

    main()
