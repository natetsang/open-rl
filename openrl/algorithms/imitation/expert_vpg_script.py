"""
VPG using shared NN model.
"""
import gym
import time
import pickle
import argparse
import numpy as np
import tensorflow as tf
from typing import Union, List, Callable, Tuple
from models.models import actor_critic_fc_discrete_network
from util.compute_returns import compute_returns_simple

# Set up constants
GAMMA = 0.99
LEARNING_RATE = 0.001
ACTOR_LOSS_WEIGHT = 1.0
CRITIC_LOSS_WEIGHT = 0.1

# Expert constants
EXPERT_POLICY_SAVE_PATH = "./checkpoints/expert_model_weights"
EXPERT_POLICY_TRAINING_REWARD_THRESHOLD = 120
EXPERT_DATASET_SIZE_LIM = 2000
EXPERT_DATASET_SAVE_PATH = "expert_data.pkl"


class VPGAgent:
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

        # Save directories
        self.save_dir = save_dir

    def save_models(self) -> None:
        # TODO - I'm saving the weights b/c I was getting an error doing the whole model
        #   But I should eventually see if the newer version won't cause an error!
        self.model.save_weights(self.save_dir)

    def load_models(self) -> tf.keras.Model:
        self.model = tf.keras.models.load_model(self.save_dir)
        return self.model

    def train_episode(self) -> Tuple[Union[float, int], int]:
        ep_rewards = 0
        state = env.reset()
        done = False
        cur_step = 0
        reward_trajectory, state_trajectory, action_prob_trajectory = [], [], []
        value_trajectory = []
        with tf.GradientTape() as tape:
            # Rollout policy to get a single trajectory
            while not done:
                cur_step += 1
                # Get state in correct format
                state = tf.expand_dims(tf.convert_to_tensor(state), 0)
                state_trajectory.append(state)

                # Predict action prob and take action
                action_prob, values = self.model(state)
                action = np.random.choice(self.num_actions, p=np.squeeze(action_prob))

                state, reward, done, _ = env.step(action)

                # Some bookkeeping
                ep_rewards += reward
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
            actor_loss = -tf.math.log(action_prob_trajectory) * tf.stop_gradient(advantages)
            critic_loss = tf.square(advantages)
            total_loss = tf.reduce_sum(actor_loss * ACTOR_LOSS_WEIGHT +
                                       critic_loss * CRITIC_LOSS_WEIGHT)

        # Backpropagate loss
        grads = tape.gradient(total_loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return ep_rewards, cur_step

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

    def sample_trajectory(self) -> List[Tuple]:
        traj = []
        state = self.env.reset()
        done = False

        while not done:
            # Select action
            action_prob, _ = self.model(tf.expand_dims(state, axis=0))
            action = np.argmax(np.squeeze(action_prob))

            # Interact with environment
            next_state, reward, done, _ = env.step(action)

            traj.append((state, action, reward, next_state, done))
            state = next_state
        return traj


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v0")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()

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
    agent = VPGAgent(environment=env,
                     model_fn=actor_critic_fc_discrete_network,
                     optimizer=opt,
                     model_kwargs=dict(state_dims=_state_dims,
                                       num_hidden_layers=2,
                                       hidden_size=128,
                                       num_actions=_num_actions),
                     save_dir=EXPERT_POLICY_SAVE_PATH)

    # Run training
    best_mean_rewards = -1e8
    running_reward = 0
    start = time.time()
    for e in range(args.epochs):
        ep_rew, ep_steps = agent.train_episode()
        running_reward = 0.05 * ep_rew + (1 - 0.05) * running_reward

        if e % 10 == 0:
            template = "running reward: {:.2f} | episode reward: {:.2f} at episode {}"
            print(template.format(running_reward, ep_rew, e))

        if running_reward > EXPERT_POLICY_TRAINING_REWARD_THRESHOLD:
            print("Solved at episode {}!".format(e))
            break

    # Now that we've completed training, let's plot the results
    print(f"Training time elapsed (sec): {round(time.time() - start, 2)}")

    print("Generating expert dataset with trained policy!")
    transitions = []
    while len(transitions) < EXPERT_DATASET_SIZE_LIM:
        tr = agent.sample_trajectory()
        transitions.extend(tr)

    print(f"Saving dataset to {EXPERT_DATASET_SAVE_PATH}")
    with open(EXPERT_DATASET_SAVE_PATH, 'wb') as file:
        pickle.dump(transitions, file)

    # Save expert policy
    print(f"Saving expert policy model to {EXPERT_POLICY_SAVE_PATH}")
    agent.save_models()
