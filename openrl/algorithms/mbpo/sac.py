import gym
import numpy as np
import tensorflow as tf
from typing import Union, Callable, Tuple
from util.replay_buffer import ReplayBuffer
import tensorflow_probability as tfp


# Set up
GAMMA = 0.99


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
        self.train_batch_size = train_kwargs.get("train_batch_size")

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

    def train_episode(self):
        # Retrieve batch of transitions
        transitions = self.replay_buffer.sample(batch_size=self.train_batch_size)

        if transitions is None:
            return 0., 0., 0.

        states, actions, rewards, next_states, dones = transitions

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
            critic_loss1 = tf.reduce_mean(tf.square(targets - q_values)) * 0.5  # Some of examples apply this factor
        grads = tape.gradient(critic_loss1, self.critic_model1.trainable_variables)
        self.critic1_optimizer.apply_gradients(zip(grads, self.critic_model1.trainable_variables))

        # Calculate critic2 loss
        with tf.GradientTape() as tape:
            q_values = self.critic_model2([states, actions])
            critic_loss2 = tf.reduce_mean(tf.square(targets - q_values)) * 0.5  # Some of examples apply this factor
        grads = tape.gradient(critic_loss2, self.critic_model2.trainable_variables)
        self.critic2_optimizer.apply_gradients(zip(grads, self.critic_model2.trainable_variables))

        # Delay policy by only updating it every d iterations
        # Step 5: Calculate actor loss and do gradient step
        actor_loss = 0
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

        self.cur_episode += 1

        return actor_loss, critic_loss1, critic_loss2

    def run_agent(self, render=False) -> Tuple[float, int]:
        total_reward, total_steps = 0, 0
        state = self.env.reset()
        done = False

        while not done:
            if render:
                self.env.render()

            # Select action
            action, _ = self.actor_model(tf.expand_dims(state, axis=0))

            # Interact with environment
            state, reward, done, _ = self.env.step(action[0])

            # Bookkeeping
            total_reward += reward
            total_steps += 1
        return total_reward, total_steps
