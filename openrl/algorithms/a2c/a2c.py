"""
N-step A2C using shared NN model with GAE and entropy.
"""
import gym
import time
import argparse
import numpy as np
import tensorflow as tf
from typing import Union, List, Callable, Tuple
from multiprocessing_env import SubprocVecEnv
from .models import actor_critic_fc_discrete_network
from .utils import plot_training_results


# Set up
GAMMA = 0.99
LAMBDA = 0.95
LEARNING_RATE = 0.001
ACTOR_LOSS_WEIGHT = 1.0
CRITIC_LOSS_WEIGHT = 0.5
ENTROPY_LOSS_WEIGHT = 0.01
MAX_STEPS_PER_ENV = 256 * 3  # The number of steps each env takes per epoch
TEST_FREQ = 1  # Evaluate the agent at this cadence

debug = False
if debug:
    NUM_ENVS = 3
    MAX_STEPS_PER_ENV = 2
    N_STEPS = 2
    EPOCHS = 1
    TEST_EPOCHS = 1


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


class ActorCriticAgent:
    def __init__(self,
                 environments: SubprocVecEnv,
                 eval_env: gym.Env,
                 model_fn: Callable[..., tf.keras.Model],
                 optimizer: tf.keras.optimizers,
                 model_kwargs: dict = None,
                 train_kwargs: dict = None,
                 save_dir: str = None) -> None:
        # Env vars
        self.envs = environments
        self.eval_env = eval_env
        self.num_inputs = model_kwargs.get('num_inputs')
        self.num_actions = model_kwargs.get('num_actions')

        # Model vars
        self.model = model_fn(num_inputs=self.num_inputs,
                              num_hidden_layers=model_kwargs.get("num_hidden_layers"),
                              hidden_size=model_kwargs.get("hidden_size"),
                              num_actions=self.num_actions)
        self.optimizer = optimizer

        # Training vars
        self.max_steps_per_env = train_kwargs.get("max_steps_per_env", 256 * 2)
        self.n_steps = train_kwargs.get("n_steps", 10)
        self.use_gae = train_kwargs.get("use_gae", True)

        self.save_dir = save_dir

    def save_model(self) -> None:
        self.model.save(self.save_dir)

    def load_model(self) -> None:
        self.model.save(self.save_dir)

    def train_episode(self):
        # Calculate number of gradient updates per episode
        num_updates = self.max_steps_per_env // self.n_steps

        state = self.envs.reset()  # Reset all envs
        for i in range(num_updates):
            reward_trajectory, state_trajectory, mask_trajectory = [], [], []
            action_trajectory, prob_trajectory, action_prob_trajectory = [], [], []
            value_trajectory = []
            with tf.GradientTape() as tape:
                for _ in range(self.n_steps):
                    # Get state in correct format
                    state_trajectory.append(state)

                    # Predict action prob and take action
                    action_prob, values = self.model(state)
                    action = [np.random.choice(self.num_actions, p=np.squeeze(a_prob)) for a_prob in action_prob]

                    state, reward, done, _ = self.envs.step(action)

                    # Some bookkeeping
                    action_trajectory.append(action)
                    value_trajectory.append(values)
                    reward_trajectory.append(tf.cast(tf.reshape(reward, (args.num_envs, 1)), tf.float32))
                    mask_trajectory.append(tf.cast(tf.reshape(1 - done, (args.num_envs, 1)), tf.float32))
                    prob_trajectory.append(action_prob)
                    action_prob_trajectory.append(tf.convert_to_tensor([tf.expand_dims(action_prob[i][a], 0)
                                                                        for i, a in enumerate(action)]))

                _, next_value = self.model(state)
                returns = compute_gae_returns(next_value, reward_trajectory, mask_trajectory, value_trajectory)
                targets = compute_discounted_returns(next_value, reward_trajectory, mask_trajectory)

                # Concat
                returns = tf.concat(returns, axis=0)
                targets = tf.concat(targets, axis=0)

                prob_trajectory = tf.concat(prob_trajectory, axis=0)
                action_prob_trajectory = tf.concat(action_prob_trajectory, axis=0)
                value_trajectory = tf.concat(value_trajectory, axis=0)
                advantages = returns - value_trajectory
                # advantages = normalize(advantages)

                # Calculate losses
                actor_loss = -tf.math.log(action_prob_trajectory) * tf.stop_gradient(advantages)
                # There are different values we could use. We could use the GAE advantage or
                # instead we could just do the n-step targets - value_trajectory
                critic_loss = tf.square(advantages)
                entropy_loss = tf.reduce_sum(prob_trajectory * tf.math.log(prob_trajectory + 1e-8), axis=1)
                total_loss = tf.reduce_mean(actor_loss * ACTOR_LOSS_WEIGHT +
                                            critic_loss * CRITIC_LOSS_WEIGHT +
                                            entropy_loss * ENTROPY_LOSS_WEIGHT)

            # Backpropagate loss
            grads = tape.gradient(total_loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

    def test_agent(self, render=False) -> Tuple[Union[float, int], int]:
        total_reward = 0
        state = self.eval_env.reset()
        done = False
        cur_step = 0
        while not done:
            if render:
                self.eval_env.render()
            cur_step += 1
            action_prob, _ = self.model(tf.expand_dims(tf.convert_to_tensor(state), 0))
            action = np.argmax(np.squeeze(action_prob))
            state, reward, done, _ = self.eval_env.step(action)
            total_reward += reward
        return total_reward, cur_step


def main() -> None:
    # Make single env for testing model performance
    eval_env = gym.make(args.env)

    # Set seeds
    if args.seed:
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)
        eval_env.seed(args.seed)

    # Create helper vars for model creation
    _num_inputs = len(eval_env.observation_space.high)
    _num_actions = eval_env.action_space.n

    # Make multiple vectorized environments for synchronous training
    envs_list = [make_env(args.env) for i in range(args.num_envs)]
    envs = SubprocVecEnv(envs_list)

    # Initialize model
    opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    agent = ActorCriticAgent(environments=envs,
                             eval_env=eval_env,
                             model_fn=actor_critic_fc_discrete_network,
                             optimizer=opt,
                             model_kwargs=dict(num_inputs=_num_inputs,
                                               num_hidden_layers=1,
                                               hidden_size=128,
                                               num_actions=_num_actions),
                             train_kwargs=dict(max_steps_per_env=MAX_STEPS_PER_ENV,
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
        agent.train_episode()

        if e % TEST_FREQ == 0:
            ep_wallclock_history.append(time.time() - start)

            ep_rew, ep_steps = agent.test_agent()
            running_reward = 0.05 * ep_rew + (1 - 0.05) * running_reward

            ep_rewards_history.append(ep_rew)
            ep_running_rewards_history.append(running_reward)
            ep_steps_history.append(ep_steps)

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
                          test_freq=TEST_FREQ,
                          save_dir="./results.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v0")
    parser.add_argument("--num_envs", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=110)
    parser.add_argument("--use_gae", type=bool, default=True)
    parser.add_argument("--n_steps", type=int, default=5)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--model_checkpoint_dir", type=str, default="./model_chkpt")
    args = parser.parse_args()

    main()
