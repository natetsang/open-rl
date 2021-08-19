"""
Online actor-critic using shared NN model.

This is super unstable because it uses a single-sample batch.
Sergey Levine talks about how this is an issue in his berkeley class.

Sometimes it achieves the goal within 500-600 episodes,
but sometimes it takes like 800-900 episodes.
What looks to be happening is that the episode gets really good rewards,
but then a few bad episodes will made the running reward really bad, and
then it takes a long time to work back up.
"""

import gym
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# Configuration paramaters for the whole setup
seed = 42
gamma = 0.99  # Discount factor for past rewards
max_steps_per_episode = 10000
env = gym.make("CartPole-v0")  # Create the environment
env.seed(seed)
eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0

num_inputs = 4
num_actions = 2
def actor_critic_network(num_inputs, num_hidden, num_actions):
    inputs = layers.Input(shape=(num_inputs,))
    common = layers.Dense(num_hidden, activation="relu")(inputs)
    action = layers.Dense(num_actions, activation="softmax")(common)
    critic = layers.Dense(1)(common)
    model = keras.Model(inputs=inputs, outputs=[action, critic])
    return model

def calculate_advantage(critic_model, state, reward, next_state, done):
    """
    Target = r + gamma * V(s')
    Advantage = r + gamma * V(s') - V(s)

    Critic training:
    y_true = r  + gamma * V(s')
    y_pred = V(s)
    """

    # print("FUNCTION INPUTS")
    # print("reward_trajectory \t::: ", reward)
    # print("state_trajectory \t::: ", state)
    # print("next_state \t::: ", next_state)
    # print("done \t::: ", done)
    # print("--------")


    next_state = tf.expand_dims(tf.convert_to_tensor(next_state), 0)
    _, value_state = critic_model(state)
    _, value_next_state = critic_model(next_state)

    target = reward + gamma * value_next_state[0, 0] * (1 - int(done))
    advantage = target - value_state[0, 0]

    # print("CALCULATE DISCOUNTED REWARDS")
    # print("discounted_rewards\t::: ", target)
    # print("bootstrapped_val\t::: ", value_next_state[0, 0])
    # print("GET CRITIC VALUES")
    # print("critic_values\t::: ", value_state)
    # print("--------")
    # print("GET ADVANTAGES")
    # print("advantages\t::: ", advantage)
    # print("---------")
    return advantage


model = actor_critic_network(num_inputs, 128, num_actions)
optimizer = keras.optimizers.Adam(learning_rate=0.01)


running_reward = 0
episode_count = 0
while episode_count < 800:  # Run until solved
    state = env.reset()
    episode_reward = 0
    done = False
    while not done:
        with tf.GradientTape() as tape:
            state = tf.expand_dims(tf.convert_to_tensor(state), 0)

            action_probs, critic_value = model(state)
            action = np.random.choice(num_actions, p=np.squeeze(action_probs))

            # Apply the sampled action in our environment
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            A = calculate_advantage(model, state, reward, next_state, done)
            actor_loss = -tf.math.log(action_probs[0, action])  * A
            critic_loss = tf.square(A)
            loss_value = actor_loss + 0.5 * critic_loss

            # print("actor_loss\t::: ", actor_loss)
            # print("critic_loss\t::: ", critic_loss)
            # print("total_loss\t::: ", loss_value)
            # print("=============================================")
        grads = tape.gradient(loss_value, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        state = next_state

    # Update running reward to check condition for solving
    running_reward = 0.05 * episode_reward + (1 - 0.05) * running_reward

    # Log details
    episode_count += 1
    if episode_count % 10 == 0:
        template = "running reward: {:.2f} | episode reward: {:.2f} at episode {}"
        print(template.format(running_reward, episode_reward, episode_count))

    if running_reward > 195:  # Condition to consider the task solved
        print("Solved at episode {}!".format(episode_count))
        break
