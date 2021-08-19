"""
My first attempt at DDPG!
"""

import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import multiprocessing as mp
import threading
from queue import Queue
tf.keras.backend.set_floatx('float32')

# Set up
GAMMA = 0.99
ACTOR_LEARNING_RATE = 0.0001
CRITIC_LEARNING_RATE = 0.001
L2_REG = 0.01
NUM_EPISODES = 1

###############################################################################

class ReplayBuffer:
    def __init__(self, capacity=1000000, batch_size=64):
        self.capacity = capacity
        self.batch_size = batch_size

        self.buffer_state = np.empty(shape=(capacity, 3))
        self.buffer_action = np.empty(shape=(capacity, 1))
        self.buffer_reward = np.empty(shape=(capacity, 1))
        self.buffer_next_state = np.empty(shape=(capacity, 3))
        self.buffer_done = np.empty(shape=(capacity, 1))

        self.size = 0
        self.idx = 0

    def store_transition(self, transition):
        state, action, reward, next_state = transition
        # This will make sure to overwrite the oldest transition if full
        current_index = self.idx % self.capacity
        # print("STATE", state)
        # print("ACTION", action)
        # print("REWARD", reward)
        # print("NEXT_STATE", next_state)
        # Store transition in buffer
        self.buffer_state[current_index] = state
        self.buffer_action[current_index] = action
        self.buffer_reward[current_index] = reward
        self.buffer_next_state[current_index] = next_state

        # Increment counters
        if self.size < self.capacity:
            self.size += 1
        self.idx += 1

    def sample(self):
        # We can't sample if we don't have enough transitions
        if self.size < self.batch_size:
            return

        idxs = np.random.choice(self.size, self.batch_size)

        batch_state = self.buffer_state[idxs]
        batch_action = self.buffer_action[idxs]
        batch_reward = self.buffer_reward[idxs]
        batch_next_state = self.buffer_next_state[idxs]

        return (batch_state, batch_action, batch_reward, batch_next_state)

    def __len__(self):
        return self.size


class OUActionNoise:
    # Source: https://keras.io/examples/rl/ddpg_pendulum/
    def __init__(self, mean, std_deviation, theta=0.15, dt=1e-2, x_initial=None):
        self.theta = theta
        self.mean = mean
        self.std_dev = std_deviation
        self.dt = dt
        self.x_initial = x_initial
        self.reset()

    def __call__(self):
        x = (
            self.x_prev
            + self.theta * (self.mean - self.x_prev) * self.dt
            + self.std_dev * np.sqrt(self.dt) * np.random.normal(size=self.mean.shape)
        )
        self.x_prev = x
        return x

    def reset(self):
        if self.x_initial is not None:
            self.x_prev = self.x_initial
        else:
            self.x_prev = np.zeros_like(self.mean)


class DDPGAgent:
    def __init__(self, env):
        self.env = env
        self.env_action_ub = env.action_space.high[0]
        self.env_action_lb = env.action_space.low[0]
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.shape[0]

        self.actor = self.create_actor()
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=ACTOR_LEARNING_RATE)
        self.target_actor = self.create_actor()
        self.target_actor.set_weights(self.actor.get_weights())

        self.critic = self.create_critic()
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=CRITIC_LEARNING_RATE)
        self.target_critic = self.create_critic()
        self.target_critic.set_weights(self.critic.get_weights())

        self.tau = 0.001
        # self.replay_buffer =

    def create_actor(self):
        inputs = layers.Input(shape=(self.state_size,))

        hidden1 = layers.BatchNormalization()(inputs)
        hidden1 = layers.Dense(400, activation="relu")(hidden1)

        hidden2 = layers.BatchNormalization()(hidden1)
        hidden2 = layers.Dense(300, activation="relu")(hidden2)

        # Initialize weights of last layer [-3e-3, 3-e3]
        weight_init = tf.keras.initializers.RandomUniform(-0.003, 0.003)

        output_action = layers.BatchNormalization()(hidden2)
        output_action = layers.Dense(self.action_size, activation="tanh",
                                    kernel_initializer=weight_init)(output_action)

        # Scale outputs to be within environment action space bounds
        resize_fn = lambda x: (((x + 1) * (self.env_action_ub - self.env_action_lb)) / 2) + self.env_action_lb

        output_action_scaled = resize_fn(output_action)
        model = tf.keras.Model(inputs=inputs, outputs=output_action_scaled)
        return model

    def create_critic(self):
        state_inputs = layers.Input(shape=(self.state_size,))
        hidden_state1 = layers.BatchNormalization()(state_inputs)
        hidden_state1 = layers.Dense(400, activation="relu",
                                    kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(hidden_state1)

        action_inputs = layers.Input(shape=(self.action_size,))
        hidden_action1 = layers.BatchNormalization()(action_inputs)
        hidden_action1 = layers.Dense(400, activation="relu",
                                    kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(hidden_action1)

        hidden2 = layers.Concatenate()([hidden_state1, hidden_action1])
        hidden2 = layers.BatchNormalization()(hidden2)
        hidden2 = layers.Dense(300, activation="relu",
                              kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(hidden2)

        # Initialize weights of last layer [-3e-3, 3-e3]
        weight_init = tf.keras.initializers.RandomUniform(-0.003, 0.003)

        output = layers.BatchNormalization()(hidden2)
        output = layers.Dense(1, kernel_initializer=weight_init,
                             kernel_regularizer=tf.keras.regularizers.l2(L2_REG))(output)

        model = tf.keras.Model(inputs=[state_inputs, action_inputs], outputs=output)
        return model

    def update_target_networks(self):
        # Actor update
        new_weights = []
        actor_weights = self.actor.get_weights()
        actor_target_weights = self.target_actor.get_weights()

        for weights, target_weights in zip(actor_weights, actor_target_weights):
            new_weights.append(self.tau * weights + (1 - self.tau) * target_weights)
        self.target_actor.set_weights(new_weights)

        # Critic update
        new_weights = []
        critic_weights = self.critic.get_weights()
        critic_target_weights = self.target_critic.get_weights()

        for weights, target_weights in zip(critic_weights, critic_target_weights):
            new_weights.append(self.tau * weights + (1 - self.tau) * target_weights)
        self.target_critic.set_weights(new_weights)

def run_episode(agent, replay_buffer, noise):
    noise.reset()
    state = tf.expand_dims(tf.convert_to_tensor(agent.env.reset()), 0)
    ep_reward = 0
    done = False
    while not done:
        # print("NEXT STATE::::: ", next_state)
        # state = tf.expand_dims(tf.convert_to_tensor(next_state), 0)
        # print("STATE:::::  ", state)
        action = agent.actor(state) + noise()
        action = tf.clip_by_value(action, agent.env_action_lb, agent.env_action_ub)

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
        states, actions, rewards, next_states = batch_transitions

        # Calculate critic loss
        with tf.GradientTape() as tape:
            target_next_actions = agent.target_actor(next_states)
            ys = rewards + GAMMA * tf.stop_gradient(agent.target_critic([next_states, target_next_actions])) # added this..makes sense to me but need to confirm
            Qs = agent.critic([states, actions])
            critic_loss = tf.reduce_mean(tf.square(ys - Qs))  # Removed tf.stop_gradient(Qs) because it was giving error
        grads = tape.gradient(critic_loss, agent.critic.trainable_variables)
        agent.critic_optimizer.apply_gradients(zip(grads, agent.critic.trainable_variables))

        # Calculate actor loss
        with tf.GradientTape() as tape:
            actor_actions = agent.actor(states)
            critic_values = agent.critic([states, actor_actions])
            actor_loss = -tf.reduce_mean(critic_values)
        grads = tape.gradient(actor_loss, agent.actor.trainable_variables)
        agent.actor_optimizer.apply_gradients(zip(grads, agent.actor.trainable_variables))

        # "slow" update of target weights
        agent.update_target_networks()
        state = next_state
    return ep_reward


if __name__ == '__main__':
    env = gym.make("Pendulum-v0")
    agent = DDPGAgent(env)
    replay_buffer = ReplayBuffer()

    std_dev = 0.2
    noise = OUActionNoise(mean=np.zeros(1), std_deviation=float(std_dev) * np.ones(1))

    # Run training
    running_reward = 0
    for e in range(NUM_EPISODES):
        ep_rew = run_episode(agent, replay_buffer, noise)

        running_reward = 0.05 * ep_rew + (1 - 0.05) * running_reward

        if e % 10 == 0:
            template = "running reward: {:.2f} | episode reward: {:.2f} at episode {}"
            print(template.format(running_reward, ep_rew, e))

        if running_reward > 195:
            print("Solved at episode {}!".format(e))
            break
