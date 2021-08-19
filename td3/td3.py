"""
My first attempt at DDPG!
"""

import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

tf.keras.backend.set_floatx('float32')

# Set up
GAMMA = 0.99
ACTOR_LEARNING_RATE = 3e-4
CRITIC_LEARNING_RATE = 3e-4
L2_REG = 0.01
NUM_EPISODES = 101

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


class TD3Agent:
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

        # Twin 1
        self.critic1 = self.create_critic()
        self.target_critic1 = self.create_critic()
        self.target_critic1.set_weights(self.critic1.get_weights())
        # Twin 2
        self.critic2 = self.create_critic()
        self.target_critic2 = self.create_critic()
        self.target_critic2.set_weights(self.critic2.get_weights())

        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=CRITIC_LEARNING_RATE)

        self.tau = 0.005
        self.total_it = 0
        self.policy_freq = 2

        self.noise_clip = 0.5
        self.policy_noise = 0.2


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
        inputs_state = layers.Input(shape=(self.state_size,))
        inputs_action = layers.Input(shape=(self.action_size,))
        inputs_concat = layers.concatenate([inputs_state, inputs_action])
        hidden1 = layers.Dense(400, activation="relu")(inputs_concat)
        hidden2 = layers.Dense(300, activation="relu")(hidden1)
        output = layers.Dense(1)(hidden2)

        model = tf.keras.Model(inputs=[inputs_state, inputs_action], outputs=output)
        return model

    def update_target_networks(self):
        # Actor update
        new_weights = []
        actor_weights = self.actor.get_weights()
        actor_target_weights = self.target_actor.get_weights()

        for weights, target_weights in zip(actor_weights, actor_target_weights):
            new_weights.append(self.tau * weights + (1 - self.tau) * target_weights)
        self.target_actor.set_weights(new_weights)

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
        # print("NEXT STATE::::: ", next_state)
        # state = tf.expand_dims(tf.convert_to_tensor(next_state), 0)
        # print("STATE:::::  ", state)
        action = agent.actor(state)  + tf.random.normal(agent.env.action_space.shape) * agent.policy_noise
        action = tf.clip_by_value(action,
                                    clip_value_min=agent.env_action_lb,
                                    clip_value_max=agent.env_action_ub)
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

        # ADD NOISE
        noise = tf.clip_by_value(tf.random.normal(agent.env.action_space.shape) * agent.policy_noise,
                                clip_value_min=-agent.noise_clip, clip_value_max=agent.noise_clip)

        target_next_actions = tf.clip_by_value(agent.target_actor(next_states) + noise,
                                clip_value_min=agent.env_action_lb,
                                clip_value_max=agent.env_action_ub)

        Qtarget1 = agent.target_critic1([next_states, target_next_actions])
        Qtarget2 = agent.target_critic2([next_states, target_next_actions])
        Qtarget_min = tf.math.minimum(Qtarget1, Qtarget2)
        ys = rewards + GAMMA * Qtarget_min

        # Calculate critic1 loss
        with tf.GradientTape() as tape:
            Qs = agent.critic1([states, actions])
            critic_loss = tf.reduce_mean(tf.square(ys - Qs))
        grads = tape.gradient(critic_loss, agent.critic1.trainable_variables)
        agent.critic_optimizer.apply_gradients(zip(grads, agent.critic1.trainable_variables))

        # Calculate critic2 loss
        with tf.GradientTape() as tape:
            Qs = agent.critic2([states, actions])
            critic_loss = tf.reduce_mean(tf.square(ys - Qs))
        grads = tape.gradient(critic_loss, agent.critic2.trainable_variables)
        agent.critic_optimizer.apply_gradients(zip(grads, agent.critic2.trainable_variables))

        if agent.total_it % agent.policy_freq == 0:
        # Calculate actor loss
            with tf.GradientTape() as tape:
                actor_actions = agent.actor(states)
                critic_values = agent.critic1([states, actor_actions])
                actor_loss = -tf.reduce_mean(critic_values)
            grads = tape.gradient(actor_loss, agent.actor.trainable_variables)
            agent.actor_optimizer.apply_gradients(zip(grads, agent.actor.trainable_variables))

            # "slow" update of target weights
            agent.update_target_networks()

        state = next_state
    return ep_reward


if __name__ == '__main__':
    env = gym.make("Pendulum-v0")
    agent = TD3Agent(env)
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
