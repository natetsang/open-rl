"""
A3C using multithreading!
"""

import gym
import time
import argparse
import numpy as np
import tensorflow as tf
from typing import Union, List, Callable, Tuple
from models.models import actor_critic_fc_discrete_network
from .utils import plot_training_results

import multiprocessing as mp
import threading
from queue import Queue


# Set up
GAMMA = 0.99
LAMBDA = 0.95
LEARNING_RATE = 0.001
ACTOR_LOSS_WEIGHT = 1.0
CRITIC_LOSS_WEIGHT = 0.5
ENTROPY_LOSS_WEIGHT = 0.01
TEST_FREQ = 1  # Evaluate the agent at this cadence


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

        # Notice I'm adding back the value to get the reward
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


class ActorCriticWorker(threading.Thread):
    # Class variables that all workers share
    global_max_epochs: int = 500
    global_current_epoch: int = 0
    global_running_reward: Union[int, float] = 0
    global_best_score: Union[int, float] = 0
    save_lock = threading.Lock()

    train_rewards_history: List = []
    train_steps_history: List = []
    train_running_rewards_history: List = []
    train_wallclock_history: List = []
    train_wallclock_start_time: float

    def __init__(self,
                 environment: gym.Env,
                 model_fn: Callable[..., tf.keras.Model],
                 optimizer: tf.keras.optimizers,
                 model_kwargs: dict = None,
                 train_kwargs: dict = None,
                 result_queue: Queue = None,
                 global_worker: 'ActorCriticWorker' = None,
                 save_dir: str = None) -> None:
        super(ActorCriticWorker, self).__init__()
        self.result_queue = result_queue

        # Env vars
        self.env = environment
        self.num_inputs = model_kwargs.get('num_inputs')
        self.num_actions = model_kwargs.get('num_actions')

        # Local model vars
        self.local_model = model_fn(state_dims=self.num_inputs,
                                    num_actions=self.num_actions,
                                    num_hidden_layers=model_kwargs.get("num_hidden_layers"),
                                    hidden_size=model_kwargs.get("hidden_size"))
        self.local_optimizer = optimizer

        # Local training vars
        if train_kwargs and train_kwargs.get("num_epochs"):
            ActorCriticWorker.global_max_epochs = train_kwargs.get("num_epochs")
        self.max_steps_per_epoch = train_kwargs.get("max_steps_per_epoch", 1000) if train_kwargs else 0
        self.n_steps = train_kwargs.get("n_steps", 10) if train_kwargs else 0
        self.use_gae = train_kwargs.get("use_gae", True) if train_kwargs else 0

        # Global model vars
        self.global_model = global_worker.local_model if global_worker else None
        self.global_optimizer = global_worker.local_optimizer if global_worker else None

        # Save
        self.save_dir = save_dir

    def save_model(self) -> None:
        self.local_model.save(self.save_dir)

    def load_model(self) -> tf.keras.Model:
        self.local_model = tf.keras.models.load_model(self.save_dir)
        return self.local_model

    def run(self) -> None:
        # Run training
        while ActorCriticWorker.global_current_epoch < ActorCriticWorker.global_max_epochs:
            ep_rewards, ep_steps = self.train_episode()

            # TODO >> I could instead do `if len(ActorCriticWorker.train_rewards_history) == 0:
            if self.result_queue.empty():
                ActorCriticWorker.global_running_reward = ep_rewards
            else:
                ActorCriticWorker.global_running_reward = (
                        0.05 * ep_rewards + (1 - 0.05) * ActorCriticWorker.global_running_reward
                )

            self.result_queue.put(ActorCriticWorker.global_running_reward)  # TODO >> I don't think I need this

            if ActorCriticWorker.global_current_epoch % TEST_FREQ == 0:
                ActorCriticWorker.train_wallclock_history.append(
                    time.time() - ActorCriticWorker.train_wallclock_start_time)
                ActorCriticWorker.train_rewards_history.append(ep_rewards)
                ActorCriticWorker.train_running_rewards_history.append(ActorCriticWorker.global_running_reward)
                ActorCriticWorker.train_steps_history.append(ep_steps)

            if ActorCriticWorker.global_current_epoch % 10 == 0:
                template = "running reward: {:.2f} | episode reward: {:.2f} | global episode {}"
                print(template.format(ActorCriticWorker.global_running_reward, ep_rewards,
                                      ActorCriticWorker.global_current_epoch))

            if ActorCriticWorker.global_running_reward > 195:
                print("Solved at episode {}!".format(ActorCriticWorker.global_current_epoch))
                break

            with ActorCriticWorker.save_lock:
                ActorCriticWorker.global_current_epoch += 1

                if ep_rewards > ActorCriticWorker.global_best_score:
                    ActorCriticWorker.global_best_score = ep_rewards

        # If we have hit the total episode limit, end
        self.result_queue.put(None)

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
                    action_prob, values = self.local_model(state)
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

                _, next_value = self.local_model(tf.expand_dims(tf.convert_to_tensor(state), 0))
                returns = compute_gae_returns(next_value, reward_trajectory, mask_trajectory, value_trajectory)
                targets = compute_discounted_returns(next_value, reward_trajectory, mask_trajectory)

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
            grads = tape.gradient(total_loss, self.local_model.trainable_variables)
            self.global_optimizer.apply_gradients(zip(grads, self.global_model.trainable_variables))
            self.local_model.set_weights(self.global_model.get_weights())

        return ep_rewards, total_steps

    def test_agent(self, render=False) -> Tuple[Union[float, int], int]:
        total_reward = 0
        state = self.env.reset()
        done = False
        cur_step = 0
        while not done:
            if render:
                self.env.render()
            cur_step += 1
            action_prob, _ = self.local_model(tf.expand_dims(tf.convert_to_tensor(state), 0))
            action = np.argmax(np.squeeze(action_prob))
            state, reward, done, _ = self.env.step(action)
            total_reward += reward
        return total_reward, cur_step


def main() -> None:
    # Create queue
    res_queue = Queue()

    # Create environment
    env = gym.make(args.env)

    # Set seeds
    if args.seed:
        np.random.seed(args.seed)
        tf.random.set_seed(args.seed)
        env.seed(args.seed)

    # Create helper vars for model creation
    _num_inputs = len(env.observation_space.high)
    _num_actions = env.action_space.n

    # Create master agent
    opt = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    global_agent = ActorCriticWorker(environment=env,
                                     model_fn=actor_critic_fc_discrete_network,
                                     optimizer=opt,
                                     model_kwargs=dict(num_inputs=_num_inputs,
                                                       num_hidden_layers=2,
                                                       hidden_size=128,
                                                       num_actions=_num_actions),
                                     save_dir=args.model_checkpoint_dir)

    # Create list of workers
    workers = [
        ActorCriticWorker(
            environment=gym.make(args.env),
            model_fn=actor_critic_fc_discrete_network,
            optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
            model_kwargs=dict(num_inputs=_num_inputs,
                              num_hidden_layers=2,
                              hidden_size=128,
                              num_actions=_num_actions),
            train_kwargs=dict(num_epochs=args.epochs,
                              max_steps_per_epoch=1000,
                              use_gae=args.use_gae,
                              n_steps=args.n_steps),
            result_queue=res_queue,
            global_worker=global_agent,
            save_dir=args.model_checkpoint_dir
        )
        for i in range(args.num_workers)
    ]

    # Start workers
    ActorCriticWorker.train_wallclock_start_time = time.time()
    for i, w in enumerate(workers):
        print(f"Starting worker {i}!")
        w.start()

    # Wait for workers to finish before continuing main thread
    for w in workers:
        w.join()

    print("Training complete!")

    # Plot summary of results
    plot_training_results(rewards_history=ActorCriticWorker.train_rewards_history,
                          running_rewards_history=ActorCriticWorker.train_running_rewards_history,
                          steps_history=ActorCriticWorker.train_steps_history,
                          wallclock_history=ActorCriticWorker.train_wallclock_history,
                          test_freq=TEST_FREQ,
                          save_dir="./results.png")

    # TODO >> I probably can delete this since I'm now using class variables
    # moving_average_rewards = []
    # while not res_queue.empty():
    #     reward = res_queue.get()
    #     if reward is not None:
    #         moving_average_rewards.append(reward)
    #
    # plt.plot(moving_average_rewards)
    # plt.ylabel('Moving average ep reward')
    # plt.xlabel('Step')
    # plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v0")
    parser.add_argument("--num_workers", type=int, default=mp.cpu_count())
    parser.add_argument("--epochs", type=int, default=800)
    parser.add_argument("--use_gae", type=bool, default=True)
    parser.add_argument("--n_steps", type=int, default=10)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--model_checkpoint_dir", type=str, default="./model_chkpt")
    args = parser.parse_args()

    main()
