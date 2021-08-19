"""
N-step actor-critic using shared NN model with GAE.
"""

import gym
import numpy as np
import scipy.signal
import tensorflow as tf
from tensorflow.keras import layers

# Set up
env = gym.make("CartPole-v0")
GAMMA = 0.99
LAMBDA = 0.95
LEARNING_RATE = 0.001
ENTROPY_WEIGHT = 0.01

num_inputs = len(env.observation_space.high)
num_actions = env.action_space.n

# For repeatable experiments
# seed = 42
# np.random.seed(seed)
# tf.random.set_seed(seed)
# env.seed(seed)

def actor_critic_network(num_inputs, num_hidden, num_actions):
    inputs = layers.Input(shape=(num_inputs,), name="Input_layer")
    common = layers.Dense(num_hidden, activation="relu", name="Common_hidden_layer")(inputs)
    action = layers.Dense(num_actions, activation="softmax", name="Action_output_layer")(common)
    critic = layers.Dense(1, name="Critic_output_layer")(common)
    model = tf.keras.Model(inputs=inputs, outputs=[action, critic])
    return model

def calculate_advantages(model, reward_history, state_history, next_state, reached_done, gamma=GAMMA):
    # Get bootstrapped value for the n+1 step
    _, bootstrapped_value = model(tf.expand_dims(tf.convert_to_tensor(next_state, dtype=tf.float32), 0))
    bootstrapped_value = tf.reshape(bootstrapped_value, [1])
    final_value = tf.expand_dims(0.0, 0) if reached_done else bootstrapped_value

    # Calculate critic values V(s) for each state in the trajectory
    _, critic_values = model(tf.convert_to_tensor(np.vstack(state_history), dtype=tf.float32))
    critic_values = tf.reshape(critic_values, [critic_values.shape[0]])
    critic_values = tf.concat([critic_values, final_value], 0)

    # Calculate advantages
    # Originally I was using scipy.signal.lfilter like in OpenAI Spinning up,
    # but tensorflow wasn't able to recognize it in the graph, so it was causing issues
    # To rectify this, I've replaced that with a manual calculation below
    deltas = reward_history + gamma * critic_values[1:] - critic_values[:-1]
    advantages = []
    gae = 0
    for d in deltas[::-1]:
        gae = d + GAMMA * LAMBDA * gae
        advantages.append(gae)
    advantages.reverse()

    # Normalize discounted rewards if n-step > 1
    # Commented out because empirically it doesn't seem to converge when used!!
    # if len(advantages) > 1:
        # advantages = (advantages - advantages.mean()) / np.maximum(advantages.std(), 1e-6)
    return tf.cast(tf.convert_to_tensor(advantages), dtype=tf.float32)

def run_episode(n_steps):
    ep_rewards = 0
    state = env.reset()
    done = False
    while not done:
        cur_step = 0
        reward_trajectory, state_trajectory = [], []
        prob_trajectory, action_prob_trajectory = [], []
        with tf.GradientTape() as tape:
            while (cur_step < n_steps and not done):
                cur_step += 1

                # Get state in correct format
                state = tf.expand_dims(tf.convert_to_tensor(state), 0)
                state_trajectory.append(state)

                # Predict action prob and take action
                action_prob, _ = model(state)
                action = np.random.choice(num_actions, p=np.squeeze(action_prob))
                state, reward, done, _ = env.step(action)

                # Some bookkeeping
                ep_rewards += reward
                reward_trajectory.append(reward)
                prob_trajectory.append(action_prob)
                action_prob_trajectory.append(action_prob[0, action])

            # Calculate entropies
            prob_trajectory = tf.convert_to_tensor(np.vstack(prob_trajectory), dtype=tf.float32)
            entropies = tf.reduce_sum(prob_trajectory * tf.math.log(prob_trajectory + 1e-8), axis=1)

            # Calculate advantages
            A = calculate_advantages(model, reward_trajectory, state_trajectory, state, done)

            # Calculate losses
            actor_loss = -tf.math.log(tf.convert_to_tensor(action_prob_trajectory)) * tf.stop_gradient(A)
            actor_loss += ENTROPY_WEIGHT * entropies
            critic_loss = tf.square(A)
            total_loss = tf.reduce_mean(actor_loss + 0.5 * critic_loss)

        # Backpropagate loss
        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return ep_rewards


if __name__ == '__main__':
    # Initialize model
    model = actor_critic_network(num_inputs, 128, num_actions)
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    # Run training
    running_reward = 0
    for e in range(800):
        ep_rew = run_episode(n_steps=5)

        running_reward = 0.05 * ep_rew + (1 - 0.05) * running_reward

        if e % 10 == 0:
            template = "running reward: {:.2f} | episode reward: {:.2f} at episode {}"
            print(template.format(running_reward, ep_rew, e))

        if running_reward > 195:
            print("Solved at episode {}!".format(e))
            break
