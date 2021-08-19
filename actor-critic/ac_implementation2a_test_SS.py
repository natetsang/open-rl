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
env = gym.make("CartPole-v0")  # Create the environment
env.seed(seed)
eps = np.finfo(np.float32).eps.item()  # Smallest number such that 1.0 + eps != 1.0

num_inputs = 4
num_actions = 2
def actor_critic_network(num_inputs, num_hidden, num_actions):
    inputs = layers.Input(shape=(num_inputs,), name="Input_layer")
    common = layers.Dense(num_hidden, activation="relu", name="Common_hidden_layer")(inputs)
    action = layers.Dense(num_actions, activation="softmax", name="Action_output_layer")(common)
    critic = layers.Dense(1, name="Critic_output_layer")(common)

    model = tf.keras.Model(inputs=inputs, outputs=[action, critic])
    return model

def calculate_advantages(model, reward_history, state_history, next_state, reached_done, gamma=0.99):
    # print("FUNCTION INPUTS")
    # print("reward_trajectory \t::: ", reward_history)
    # print("state_trajectory \t::: ", state_history)
    # print("next_state \t::: ", next_state)
    # print("done \t::: ", reached_done)
    # print("--------")

    ####################### GET DISCOUNTED REWARDS ############################
    # print("CALCULATE DISCOUNTED REWARDS")
    _, bootstrapped_value = model(tf.expand_dims(tf.convert_to_tensor(next_state, dtype=tf.float32), 0))
    bootstrapped_value = tf.squeeze(bootstrapped_value)  #.numpy()[0]
    # print("bootstrapped_val\t::: ", bootstrapped_value)

    # Vs = tf.convert_to_tensor(0, dtype=tf.float32) if reached_done else bootstrapped_value
    # print("Vs\t::: ", Vs)
    # discounted_rewards = reward_history + gamma * Vs
    # discounted_rewards = tf.squeeze(discounted_rewards)  # tf.Tensor(x)
    # discounted_rewards = tf.convert_to_tensor(discounted_rewards)
    discounted_rewards = []
    total_ret = 0.0 if reached_done else bootstrapped_value
    for r in reward_history[::-1]:
        total_ret = r + gamma * total_ret
        discounted_rewards.append(total_ret)
    discounted_rewards.reverse()
    discounted_rewards = tf.convert_to_tensor(discounted_rewards) # tf.Tensor([d1])
    # discounted_rewards = discounted_rewards - tf.keras.backend
    # print("discounted_rewards\t::: ", discounted_rewards)

    ###################### GET CRITIC VALUES #################################
    # print("GET CRITIC VALUES")
    # _, critic_values = model(state_history)
    _, critic_values = model(tf.convert_to_tensor(np.vstack(state_history)))  # input tf.Tensor[[x,x,x,x]] - output = tf.Tensor[[x]]

    # print("critic_values\t::: ", tf.squeeze(critic_values))

    # print("--------")

    ####################### GET ADVANTAGES #################################
    # print("GET ADVANTAGES")
    advantages = discounted_rewards - tf.squeeze(critic_values)
    # print("advantages\t::: ", advantages)
    # print("---------")
    return advantages


def run_episode(n_steps):
    ep_rewards = 0
    state = env.reset()
    done = False
    while not done:
        cur_step = 0
        reward_trajectory, state_trajectory, action_prob_trajectory = [], [], []
        with tf.GradientTape() as tape:
            while (cur_step < n_steps and not done):
                # Update step number
                cur_step += 1

                # Get state in correct format
                state = tf.expand_dims(tf.convert_to_tensor(state), 0)
                state_trajectory.append(state)

                # Predict action prob and take action
                action_prob, _ = model(state)
                action = np.random.choice(num_actions, p=np.squeeze(action_prob))
                state, reward, done, _ = env.step(action)
                ep_rewards += reward

                reward_trajectory.append(reward)
                action_prob_trajectory.append(action_prob[0, action])

            # Calculate losses
            A = calculate_advantages(model, reward_trajectory, state_trajectory, state, done)
            # print("action_prob_trajectory\t::: ", tf.squeeze(action_prob_trajectory))
            # print("action_prob[0, action])\t::: ", action_prob[0, action]))
            # print("action_probs_history_tf\t", tf.convert_to_tensor(action_prob_trajectory))
            # print("SQUEEZE\t", tf.squeeze(action_prob_trajectory))
            # actor_loss = -tf.math.log(tf.squeeze(action_prob_trajectory)) * tf.stop_gradient(A)
            actor_loss = -tf.math.log(tf.convert_to_tensor(action_prob_trajectory)) * tf.stop_gradient(A)
            critic_loss = tf.square(A)
            total_loss = tf.reduce_mean(actor_loss + 0.5 * critic_loss)
            # print("actor_loss\t::: ", actor_loss)
            # print("critic_loss\t::: ", critic_loss)
            # print("total_loss\t::: ", total_loss)
            # print("=============================================")

        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return ep_rewards

if __name__ == '__main__':
    # Initialize model
    model = actor_critic_network(num_inputs, 128, num_actions)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)  ## TODO >> Maybe I should change this to 0.001?

    # Run training
    running_reward = 0
    for e in range(800):
        ep_rew = run_episode(n_steps=1)

        running_reward = 0.05 * ep_rew + (1 - 0.05) * running_reward

        if e % 10 == 0:
            template = "running reward: {:.2f} | episode reward: {:.2f} at episode {}"
            print(template.format(running_reward, ep_rew, e))

        if running_reward > 195:
            print("Solved at episode {}!".format(e))
            break
