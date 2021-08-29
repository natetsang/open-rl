import gym
import time
import argparse
import numpy as np
import tensorflow as tf
from typing import Union, Callable, Tuple
from .models import sac_actor_fc_continuous_network, sac_critic_fc_continuous_network
from .utils import ReplayBuffer, plot_training_results
import tensorflow_probability as tfp
tfd = tfp.distributions


# Set up
GAMMA = 0.99
ACTOR_LEARNING_RATE = 3e-4
CRITIC_LEARNING_RATE = 3e-4
ALPHA_LEARNING_RATE = 3e-4

LOG_STD_MIN = -20
LOG_STD_MAX = 2

NUM_EPISODES = 101


class SACAgent:
    def __init__(self,
                 environment: gym.Env,
                 actor_model_fn: Callable[..., tf.keras.Model],
                 actor_optimizer: tf.keras.optimizers,
                 critic_model_fn: Callable[..., tf.keras.Model],
                 critic_optimizers: Tuple,
                 alpha_optimizer: tf.keras.optimizers,
                 replay_buffer: ReplayBuffer,
                 model_kwargs: dict = None,
                 train_kwargs: dict = None,
                 save_dir: str = None) -> None:
        # Env vars
        self.env = environment
        self.state_dims = model_kwargs.get('state_dims')
        self.action_dims = model_kwargs.get('action_dims')
        self.env_action_lb = self.env.action_space.low[0]
        self.env_action_ub = self.env.action_space.high[0]

        num_hidden_layers = model_kwargs.get('num_hidden_layers')
        hidden_size = model_kwargs.get('hidden_size')

        # Actor and target actor models
        self.actor_model = actor_model_fn(state_dims=self.state_dims,
                                          action_dims=self.action_dims,
                                          env_action_lb=self.env_action_lb,
                                          env_action_ub=self.env_action_ub,
                                          log_std_min=model_kwargs.get('log_std_min'),
                                          log_std_max=model_kwargs.get('log_std_max'),
                                          num_hidden_layers=num_hidden_layers,
                                          hidden_size=hidden_size)
        self.target_actor_model = actor_model_fn(state_dims=self.state_dims,
                                                 action_dims=self.action_dims,
                                                 env_action_lb=self.env_action_lb,
                                                 env_action_ub=self.env_action_ub,
                                                 log_std_min=model_kwargs.get('log_std_min'),
                                                 log_std_max=model_kwargs.get('log_std_max'),
                                                 num_hidden_layers=num_hidden_layers,
                                                 hidden_size=hidden_size)
        self.target_actor_model.set_weights(self.actor_model.get_weights())
        self.actor_optimizer = actor_optimizer

        # Twin 1 - Critic and target critic models
        self.critic_model1 = critic_model_fn(state_dims=self.state_dims,
                                             action_dims=self.action_dims,
                                             num_hidden_layers=num_hidden_layers,
                                             hidden_size=hidden_size)
        self.critic1_optimizer = critic_optimizers[0]
        self.target_critic_model1 = critic_model_fn(state_dims=self.state_dims,
                                                    action_dims=self.action_dims,
                                                    num_hidden_layers=num_hidden_layers,
                                                    hidden_size=hidden_size)
        self.target_critic_model1.set_weights(self.critic_model1.get_weights())

        # Twin 2 - Critic and target critic models
        self.critic_model2 = critic_model_fn(state_dims=self.state_dims,
                                             action_dims=self.action_dims,
                                             num_hidden_layers=num_hidden_layers,
                                             hidden_size=hidden_size)
        self.critic2_optimizer = critic_optimizers[1]
        self.target_critic_model2 = critic_model_fn(state_dims=self.state_dims,
                                                    action_dims=self.action_dims,
                                                    num_hidden_layers=num_hidden_layers,
                                                    hidden_size=hidden_size)
        self.target_critic_model2.set_weights(self.critic_model2.get_weights())

        # Entropy temperature
        self.log_alpha = tf.Variable(0.0)
        self.alpha = tfp.util.DeferredTensor(self.log_alpha, tf.math.exp)

        self.target_entropy = -np.prod(self.env.action_space.shape)
        self.alpha_optimizer = alpha_optimizer

        # Replay buffer
        self.replay_buffer = replay_buffer

        # Training vars
        self.cur_episode = 0
        self.policy_update_freq = train_kwargs.get("policy_update_freq", 2)

        # Other vars
        self.tau = 0.005

        # Save directories
        if save_dir:
            self.save_dir_actor = save_dir + "_actor"
            self.save_dir_critic = save_dir + "_critic"

    def save_models(self) -> None:
        if self.save_dir_actor and self.save_dir_critic:
            self.actor_model.save(self.save_dir_actor)
            self.critic_model1.save(self.save_dir_critic)
            self.target_actor_model.save(self.save_dir_actor + "_target")
            self.target_critic_model1.save(self.save_dir_critic + "_target")

    def load_models(self) -> Union[Tuple[tf.keras.Model, tf.keras.Model, tf.keras.Model, tf.keras.Model], None]:
        if self.save_dir_actor and self.save_dir_critic:
            self.actor_model = tf.keras.models.load_model(self.save_dir_actor)
            self.critic_model1 = tf.keras.models.load_model(self.save_dir_critic)
            self.target_actor_model = tf.keras.models.load_model(self.save_dir_actor + "_target")
            self.target_critic_model1 = tf.keras.models.load_model(self.save_dir_critic + "_target")
            return self.actor_model, self.target_actor_model, self.critic_model1, self.target_critic_model1

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

        # Twin 1 critic update
        new_weights = []
        critic_weights = self.critic_model1.get_weights()
        target_critic_weights = self.target_critic_model1.get_weights()

        for weights, target_weights in zip(critic_weights, target_critic_weights):
            new_weights.append(self.tau * weights + (1 - self.tau) * target_weights)
        self.target_critic_model1.set_weights(new_weights)

        # Twin 2 critic update
        new_weights = []
        critic_weights = self.critic_model2.get_weights()
        target_critic_weights = self.target_critic_model2.get_weights()

        for weights, target_weights in zip(critic_weights, target_critic_weights):
            new_weights.append(self.tau * weights + (1 - self.tau) * target_weights)
        self.target_critic_model2.set_weights(new_weights)

    def train_episode(self) -> Tuple[Union[float, int], int]:
        ep_rewards = 0
        state = tf.expand_dims(tf.convert_to_tensor(self.env.reset()), 0)
        done = False
        cur_step = 0

        # Rollout policy to get a single trajectory
        while not done:
            cur_step += 1
            self.cur_episode += 1

            # Get action and take step
            action, _ = self.actor_model(state)
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

            # Retrieve batch of transitions
            states, actions, rewards, next_states, dones = batch_transitions

            # Step 1: Sample target actions for next_state given current policy pi
            next_actions, logprob_next_action = self.actor_model(next_states)
            logprob_next_action = tf.expand_dims(logprob_next_action, axis=1)

            # Step 2: Get target values
            q_targets1 = self.target_critic_model1([next_states, next_actions])
            q_targets2 = self.target_critic_model2([next_states, next_actions])
            q_targets_min = tf.math.minimum(q_targets1, q_targets2)

            # Step 3: Calculate bellman backup
            targets = rewards + GAMMA * (1 - dones) * (q_targets_min - self.alpha * logprob_next_action)

            # Step 4: Calculate critic losses and do gradient step
            # Calculate critic1 loss
            with tf.GradientTape() as tape:
                q_values = self.critic_model1([states, actions])
                critic_loss = tf.reduce_mean(tf.square(targets - q_values)) * 0.5  # Some of examples apply this factor
            grads = tape.gradient(critic_loss, self.critic_model1.trainable_variables)
            self.critic1_optimizer.apply_gradients(zip(grads, self.critic_model1.trainable_variables))

            # Calculate critic2 loss
            with tf.GradientTape() as tape:
                q_values = self.critic_model2([states, actions])
                critic_loss = tf.reduce_mean(tf.square(targets - q_values)) * 0.5  # Some of examples apply this factor
            grads = tape.gradient(critic_loss, self.critic_model2.trainable_variables)
            self.critic2_optimizer.apply_gradients(zip(grads, self.critic_model2.trainable_variables))

            # Delay policy by only updating it every d iterations
            # Step 5: Calculate actor loss and do gradient step
            if self.cur_episode % self.policy_update_freq == 0:
                with tf.GradientTape() as tape:
                    actions, logprob = self.actor_model(states)
                    logprob = tf.expand_dims(logprob, axis=1)

                    # We take the min Q value!
                    critic1_values = self.critic_model1([states, actions])
                    critic2_values = self.critic_model2([states, actions])
                    critic_min = tf.math.minimum(critic1_values, critic2_values)
                    actor_loss = -tf.reduce_mean(critic_min - self.alpha * logprob)
                grads = tape.gradient(actor_loss, self.actor_model.trainable_variables)
                self.actor_optimizer.apply_gradients(zip(grads, self.actor_model.trainable_variables))

                # Step 6: "slow" update of target weights
                self.update_target_networks()

            # Step 7: Calculate alpha loss
            _, logprob = self.actor_model(states)
            logprob = tf.expand_dims(logprob, axis=1)

            with tf.GradientTape() as tape:
                alpha_loss = -tf.reduce_mean(self.log_alpha * logprob + self.target_entropy)
            grads = tape.gradient(alpha_loss, [self.log_alpha])
            self.alpha_optimizer.apply_gradients(zip(grads, [self.log_alpha]))

            state = next_state
        return ep_rewards, cur_step


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
    buffer = ReplayBuffer()

    actor_opt = tf.keras.optimizers.Adam(learning_rate=ACTOR_LEARNING_RATE)
    critic1_opt = tf.keras.optimizers.Adam(learning_rate=CRITIC_LEARNING_RATE)
    critic2_opt = tf.keras.optimizers.Adam(learning_rate=CRITIC_LEARNING_RATE)
    alpha_opt = tf.keras.optimizers.Adam(learning_rate=ALPHA_LEARNING_RATE)

    # Create agent
    agent = SACAgent(environment=env,
                     actor_model_fn=sac_actor_fc_continuous_network,
                     actor_optimizer=actor_opt,
                     critic_model_fn=sac_critic_fc_continuous_network,
                     critic_optimizers=(critic1_opt, critic2_opt),
                     alpha_optimizer=alpha_opt,
                     replay_buffer=buffer,
                     model_kwargs=dict(state_dims=_state_dims,
                                       action_dims=_action_dims,
                                       num_hidden_layers=2,
                                       hidden_size=256,
                                       log_std_min=LOG_STD_MIN,
                                       log_std_max=LOG_STD_MAX),
                     train_kwargs=dict(policy_update_freq=2),
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

        if e % 10 == 0:
            template = "running reward: {:.2f} | episode reward: {:.2f} at episode {}"
            print(template.format(running_reward, ep_rew, e))

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
                          wallclock_history=ep_wallclock_history,
                          save_dir="./results.png")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="Pendulum-v0")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--model_checkpoint_dir", type=str, default="./model_chkpt")
    args = parser.parse_args()

    main()
