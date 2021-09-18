"""
REINFORCE - very simple but now with N batches before updating
"""
import gym
import time
import argparse
import numpy as np
import tensorflow as tf
from typing import Callable, Tuple
from models.models import actor_fc_discrete_network
from algorithms.reinforce.utils import plot_training_results
from util.compute_returns import compute_returns_simple

# Set up constants
GAMMA = 0.99
LEARNING_RATE = 0.001


class REINFORCEAgent:
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
        self.batch_size = train_kwargs.get("batch_size", 1)

        # Save directories
        self.save_dir = save_dir

    def save_models(self) -> None:
        self.model.save(self.save_dir)

    def load_models(self) -> tf.keras.Model:
        self.model = tf.keras.models.load_model(self.save_dir)
        return self.model

    def train_episode(self) -> dict:
        with tf.GradientTape() as tape:
            ep_rewards = [0 for _ in range(self.batch_size)]  # TODO >> This should become a list
            ep_loss = []
            cur_step = 0
            for i in range(self.batch_size):  # TODO >> add loop over batch_size N
                state = self.env.reset()  # initialize state
                done = False
                reward_trajectory, state_trajectory, action_prob_trajectory = [], [], []

                # Rollout policy to get a single trajectory
                while not done:
                    cur_step += 1
                    # Get state in correct format
                    state = tf.expand_dims(tf.convert_to_tensor(state), 0)
                    state_trajectory.append(state)

                    # Predict action prob and take action
                    action_prob = self.model(state)
                    action = np.random.choice(self.num_actions, p=np.squeeze(action_prob))

                    state, reward, done, _ = self.env.step(action)

                    # Some bookkeeping
                    ep_rewards[i] += reward  # TODO >> maybe add to ep_rewards[i]
                    reward_trajectory.append(tf.cast(tf.reshape(reward, (1, 1)), tf.float32))
                    action_prob_trajectory.append(tf.convert_to_tensor([tf.expand_dims(action_prob[0][action], 0)]))

                # Calculate rewards

                ## Sum rewards for entire trajectory -- R(t=1 to T)
                # returns = tf.reduce_sum(reward_trajectory)

                ## Take advantage of temporal structure -- R(t=t' to T)
                returns = compute_returns_simple(rewards=reward_trajectory, gamma=GAMMA)
                returns = tf.concat(returns, axis=0)

                action_prob_trajectory = tf.concat(action_prob_trajectory, axis=0)

                # Calculate losses
                actor_loss = -tf.math.log(action_prob_trajectory) * returns
                total_loss = tf.reduce_sum(actor_loss)
                ep_loss.append(total_loss)

            # Find mean loss from batch
            mean_trajectory_loss = tf.reduce_mean(ep_loss)

        # Backpropagate loss and update policy parameters
        grads = tape.gradient(mean_trajectory_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        logs = dict()
        logs['mean_ep_rewards'] = np.mean(ep_rewards)
        logs['mean_ep_steps'] = cur_step / self.batch_size
        logs['mean_ep_total_loss'] = tf.reduce_sum(ep_loss) / self.batch_size
        return logs
        # return float(np.mean(ep_rewards)), cur_step  # TODO >> maybe we return the mean_ep_reward from the batch

    def run_agent(self, render=False) -> Tuple[float, int]:
        total_reward, total_steps = 0, 0
        state = self.env.reset()
        done = False

        while not done:
            if render:
                self.env.render()

            # Select action
            action_prob = self.model(tf.expand_dims(state, axis=0))
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
    agent = REINFORCEAgent(environment=env,
                           model_fn=actor_fc_discrete_network,
                           optimizer=opt,
                           model_kwargs=dict(state_dims=_state_dims,
                                             num_hidden_layers=2,
                                             hidden_size=128,
                                             num_actions=_num_actions),
                           train_kwargs=dict(batch_size=args.batch_size),
                           save_dir=args.model_checkpoint_dir,
                           )

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
        ep_rew = train_logs['mean_ep_rewards']
        ep_steps = train_logs['mean_ep_steps']
        ep_losses = train_logs['mean_ep_total_loss']

        running_reward = ep_rew if e == 0 else 0.05 * ep_rew + (1 - 0.05) * running_reward

        ep_rewards_history.append(ep_rew)
        ep_running_rewards_history.append(running_reward)
        ep_steps_history.append(ep_steps)
        ep_loss_history.append(ep_losses)
        ep_wallclock_history.append(time.time() - start)

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
                          loss_history=ep_loss_history,
                          wallclock_history=ep_wallclock_history,
                          save_dir="./results.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v0")
    parser.add_argument("--batch_size", type=int, default=5)  # TODO >> Add batch_size as param
    parser.add_argument("--epochs", type=int, default=400)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--model_checkpoint_dir", type=str, default="./model_chkpt")
    args = parser.parse_args()

    main()
