"""
N-step actor-critic using shared NN model.
"""

import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers

# Set up
env = gym.make("CartPole-v0")
GAMMA = 0.99
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
    bootstrapped_value = tf.squeeze(bootstrapped_value)

    # Discount rewards
    discounted_rewards = []
    total_ret = 0.0 if reached_done else bootstrapped_value
    for r in reward_history[::-1]:
        total_ret = r + gamma * total_ret
        discounted_rewards.append(total_ret)
    discounted_rewards.reverse()
    discounted_rewards = tf.convert_to_tensor(discounted_rewards)

    # Normalize discounted rewards if n-step > 1
    # Commented out because empirically it doesn't seem to converge when used!!
    # if len(discounted_rewards) > 1:
    #     discounted_rewards -= np.mean(discounted_rewards)
    #     discounted_rewards /= (np.std(discounted_rewards) + 1e-8)

    # Calculate critic values V(s) for each state in the trajectory
    _, critic_values = model(tf.convert_to_tensor(np.vstack(state_history), dtype=tf.float32))

    # Calculate advantages
    advantages = discounted_rewards - tf.squeeze(critic_values)
    return advantages

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
        ep_rew = run_episode(n_steps=10)

        running_reward = 0.05 * ep_rew + (1 - 0.05) * running_reward

        if e % 10 == 0:
            template = "running reward: {:.2f} | episode reward: {:.2f} at episode {}"
            print(template.format(running_reward, ep_rew, e))

        if running_reward > 195:
            print("Solved at episode {}!".format(e))
            break
