"""
My first attempt at SAC!
"""

import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_probability as tfp
tfd = tfp.distributions

from replay_buffer import ReplayBuffer

tf.keras.backend.set_floatx('float32')

# Set up
GAMMA = 0.99
ACTOR_LEARNING_RATE = 3e-4
CRITIC_LEARNING_RATE = 3e-4
ALPHA_LEARNING_RATE = 3e-4

LOG_STD_MIN = -20
LOG_STD_MAX = 2

NUM_EPISODES = 101

###############################################################################

class SACAgent:
    def __init__(self, env):
        # Environment
        self.env = env
        self.env_action_ub = env.action_space.high[0]
        self.env_action_lb = env.action_space.low[0]
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.shape[0]

        # Actor
        self.actor = self.create_actor()
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=ACTOR_LEARNING_RATE)

        # Twin 1
        self.critic1 = self.create_critic()
        self.target_critic1 = self.create_critic()
        self.target_critic1.set_weights(self.critic1.get_weights())
        self.critic1_optimizer = tf.keras.optimizers.Adam(learning_rate=CRITIC_LEARNING_RATE)

        # Twin 2
        self.critic2 = self.create_critic()
        self.target_critic2 = self.create_critic()
        self.target_critic2.set_weights(self.critic2.get_weights())
        self.critic2_optimizer = tf.keras.optimizers.Adam(learning_rate=CRITIC_LEARNING_RATE)

        # Entropy temperature
        self.log_alpha = tf.Variable(0.0)
        self.alpha = tfp.util.DeferredTensor(self.log_alpha, tf.math.exp)

        self.target_entropy = -np.prod(self.env.action_space.shape)
        self.alpha_optimizer = tf.keras.optimizers.Adam(learning_rate=ALPHA_LEARNING_RATE)

        # Other variables
        self.tau = 5e-3
        self.total_it = 0
        self.policy_freq = 1


    def create_actor(self):
        inputs = layers.Input(shape=(self.state_size,))
        hidden1 = layers.Dense(256, activation="relu")(inputs)
        hidden2 = layers.Dense(256, activation="relu")(hidden1)

        # Output mean and log_std
        mu = layers.Dense(self.action_size)(hidden2)
        log_std = layers.Dense(self.action_size)(hidden2)
        log_std = tf.clip_by_value(log_std, LOG_STD_MIN, LOG_STD_MAX)

        # Create Normal distribution with outputs
        std = tf.math.exp(log_std)
        pi_dist = tfd.Normal(mu, std)

        # To obtain actions, we sample from the distribution
        # We use the reparameterization trick here
        action = pi_dist.sample()

        # Get the log probability of that action w.r.t the distribution
        logp_pi = tf.reduce_sum(pi_dist.log_prob(action), axis=1)

        # NOTE: The correction formula is a little bit magic. To get an understanding
        # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
        # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
        logp_pi -= tf.reduce_sum(2. * (tf.math.log(2.) - action - tf.math.softplus(-2. * action)), axis=1)

        # Squash the Gaussian policy
        action_squashed = tf.math.tanh(action)

        # Now either change output by multiplying by max_action,
        # action_squashed = action_squashed * self.env_action_ub

        # OR scale outputs to be within environment action space bounds
        resize_fn = lambda x: (((x + 1) * (self.env_action_ub - self.env_action_lb)) / 2) + self.env_action_lb
        action_squashed = resize_fn(action_squashed)

        model = tf.keras.Model(inputs=inputs, outputs=[action_squashed, logp_pi])
        return model

    def create_critic(self):
        inputs_state = layers.Input(shape=(self.state_size,))
        inputs_action = layers.Input(shape=(self.action_size,))
        inputs_concat = layers.concatenate([inputs_state, inputs_action])
        hidden1 = layers.Dense(256, activation="relu")(inputs_concat)
        hidden2 = layers.Dense(256, activation="relu")(hidden1)
        output = layers.Dense(1)(hidden2)

        model = tf.keras.Model(inputs=[inputs_state, inputs_action], outputs=output)
        return model

    def update_target_networks(self):
        # Critic1 update
        new_weights = []
        critic_weights = self.critic1.get_weights()
        critic_target_weights = self.target_critic1.get_weights()

        for weights, target_weights in zip(critic_weights, critic_target_weights):
            new_weights.append(self.tau * weights + (1 - self.tau) * target_weights)
        self.target_critic1.set_weights(new_weights)

        # Critic2 update
        new_weights = []
        critic_weights = self.critic2.get_weights()
        critic_target_weights = self.target_critic2.get_weights()

        for weights, target_weights in zip(critic_weights, critic_target_weights):
            new_weights.append(self.tau * weights + (1 - self.tau) * target_weights)
        self.target_critic2.set_weights(new_weights)


def run_episode(agent, replay_buffer):
    state = tf.expand_dims(tf.convert_to_tensor(agent.env.reset()), 0)
    ep_reward = 0
    done = False
    while not done:
        agent.total_it += 1

        action, _ = agent.actor(state)

        # Take step
        next_state, reward, done, _ = agent.env.step(action)
        ep_reward += reward[0]

        next_state = tf.reshape(next_state, [1, agent.state_size])
        replay_buffer.store_transition((state, action, reward, next_state))

        # Retrieve batch of transitions
        batch_transitions = replay_buffer.sample()

        # If we dont have batch_size experiences in our buffer, keep collecting samples
        if batch_transitions is None:
            state = next_state  # Clunky..Will clean up later
            continue

        # Retrieve batch of transitions
        states, actions, rewards, next_states = batch_transitions

        # Step 1: Sample target actions for next_state given current policy pi
        next_actions, logp_next_action = agent.actor(next_states)

        # Step 2: Get target values
        Qtarget1 = agent.target_critic1([next_states, next_actions])
        Qtarget2 = agent.target_critic2([next_states, next_actions])
        Qtarget_min = tf.math.minimum(Qtarget1, Qtarget2)

        # Step 3: Calculate bellman backup
        ys = rewards + GAMMA * (Qtarget_min - agent.alpha * logp_next_action)

        # Step 4: Calculate critic losses and do gradient step
        # Calculate critic1 loss
        with tf.GradientTape() as tape:
            Qs = agent.critic1([states, actions])
            critic_loss = tf.reduce_mean(tf.square(ys - Qs)) * 0.5  # Some of the examples have this factor applied
        grads = tape.gradient(critic_loss, agent.critic1.trainable_variables)
        agent.critic1_optimizer.apply_gradients(zip(grads, agent.critic1.trainable_variables))

        # Calculate critic2 loss
        with tf.GradientTape() as tape:
            Qs = agent.critic2([states, actions])
            critic_loss = tf.reduce_mean(tf.square(ys - Qs)) * 0.5
        grads = tape.gradient(critic_loss, agent.critic2.trainable_variables)
        agent.critic2_optimizer.apply_gradients(zip(grads, agent.critic2.trainable_variables))

        if agent.total_it % agent.policy_freq == 0:
        # Step 5: Calculate actor loss and do gradient step
            with tf.GradientTape() as tape:
                actions, logp_actions = agent.actor(states)

                # We take the min Q value!
                critic1_values = agent.critic1([states, actions])
                critic2_values = agent.critic2([states, actions])
                critic_min = tf.math.minimum(critic1_values, critic2_values)
                actor_loss = -tf.reduce_mean(critic_min - agent.alpha * logp_actions)
            grads = tape.gradient(actor_loss, agent.actor.trainable_variables)
            agent.actor_optimizer.apply_gradients(zip(grads, agent.actor.trainable_variables))

            # Step 6: "slow" update of target weights
            agent.update_target_networks()

        # Step 7: Calculate alpha loss
        _, logp_actions = agent.actor(states)
        with tf.GradientTape() as tape:
            alpha_loss = -tf.reduce_mean(agent.log_alpha * logp_actions + agent.target_entropy)
        grads = tape.gradient(alpha_loss, [agent.log_alpha])
        agent.alpha_optimizer.apply_gradients(zip(grads, [agent.log_alpha]))

        state = next_state
    return ep_reward


if __name__ == '__main__':
    env = gym.make("Pendulum-v0")
    agent = SACAgent(env)
    replay_buffer = ReplayBuffer()

    # Run training
    running_reward = 0
    for e in range(NUM_EPISODES):
        ep_rew = run_episode(agent, replay_buffer)

        running_reward = 0.05 * ep_rew + (1 - 0.05) * running_reward

        if e % 10 == 0:
            template = "running reward: {:.2f} | episode reward: {:.2f} at episode {}"
            print(template.format(running_reward, ep_rew, e))

        if running_reward > 195:
            print("Solved at episode {}!".format(e))
            break
