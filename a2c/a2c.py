import os, logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("tensorflow_hub").setLevel(logging.CRITICAL)

import gym
import numpy as np
import scipy.signal
import tensorflow as tf
from tensorflow.keras import layers
from multiprocessing_env import SubprocVecEnv


# Set up
NUM_ENVS = 8
GAMMA = 0.99
LAMBDA = 0.95
LEARNING_RATE = 0.001
ENTROPY_WEIGHT = 0.01
A2C_STEPS = 256  # The number of steps each env takes
N_STEPS = 10  # The number of steps each env takes per rollout
EPOCHS = 110  # The number of training rounds
TEST_EPOCHS = 1  # Evaluate the agent at this cadence

debug = False
if debug:
    NUM_ENVS = 3
    A2C_STEPS = 2
    N_STEPS = 2
    EPOCHS = 1
    TEST_EPOCHS = 1

def normalize(x):
    x -= tf.math.reduce_mean(x)
    x /= (tf.math.reduce_std(x) + 1e-8)
    return x


def make_env():
    def _thunk():
        env = gym.make("CartPole-v0")
        return env
    return _thunk


def actor_critic_network(num_inputs, num_hidden, num_actions):
    inputs = layers.Input(shape=(num_inputs,), name="Input_layer")
    common = layers.Dense(num_hidden, activation="relu", name="Common_hidden_layer")(inputs)
    action = layers.Dense(num_actions, activation="softmax", name="Action_output_layer")(common)
    critic = layers.Dense(1, name="Critic_output_layer")(common)
    model = tf.keras.Model(inputs=inputs, outputs=[action, critic])
    return model


def compute_gae(next_value, rewards, masks, values):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + GAMMA * values[step + 1] * masks[step] - values[step]
        gae = delta + GAMMA * LAMBDA * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns


def run_episode(envs, max_steps_per_env=A2C_STEPS, n_steps=N_STEPS):
    num_updates = max_steps_per_env // n_steps
    state = envs.reset()
    done = False
    for i in range(num_updates):
        # print("UPDATE #", i)
        # print("STATE: ", state)
        reward_trajectory, state_trajectory, mask_trajectory = [], [], []
        action_trajectory, prob_trajectory, action_prob_trajectory = [], [], []
        value_trajectory = []
        with tf.GradientTape() as tape:
            for _ in range(n_steps):
                # Get state in correct format
                state_trajectory.append(state)

                # Predict action prob and take action
                action_prob, values = model(state)
                # print("ACTION_PROB: ", action_prob)
                action = [np.random.choice(num_actions, p=np.squeeze(a_prob)) for a_prob in action_prob]
                # print("ACTION: ", action)

                state, reward, done, _ = envs.step(action)

                # Some bookkeeping
                action_trajectory.append(action)
                value_trajectory.append(values)
                reward_trajectory.append(tf.cast(tf.reshape(reward, (NUM_ENVS, 1)), tf.float32))
                mask_trajectory.append(tf.cast(tf.reshape(1 - done, (NUM_ENVS, 1)), tf.float32))
                prob_trajectory.append(action_prob)
                action_prob_trajectory.append(tf.convert_to_tensor([tf.expand_dims(action_prob[i][a], 0) for i, a in enumerate(action)]))
                # print("ACTION:   ", action)
                # print("PROB:     ", action_prob)
                # print("ACTION P: ", tf.convert_to_tensor([tf.expand_dims(action_prob[i][a], 0) for i, a in enumerate(action)]))
            _, next_value = model(state)
            returns = compute_gae(next_value, reward_trajectory, mask_trajectory, value_trajectory)

            # Concat
            returns = tf.concat(returns, axis=0)
            prob_trajectory = tf.concat(prob_trajectory, axis=0)
            entropies = tf.reduce_sum(prob_trajectory * tf.math.log(prob_trajectory + 1e-8), axis=1)
            action_prob_trajectory = tf.concat(action_prob_trajectory, axis=0)
            value_trajectory = tf.concat(value_trajectory, axis=0)
            advantages = returns - value_trajectory
            advantages = normalize(advantages)

            # Calculate losses
            actor_loss = -tf.math.log(action_prob_trajectory) * tf.stop_gradient(advantages)

            critic_loss = tf.square(advantages)
            total_loss = tf.reduce_mean(actor_loss + ENTROPY_WEIGHT * entropies + 0.5 * critic_loss)

        # Backpropagate loss
        grads = tape.gradient(total_loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))


def test_agent():
    total_reward = 0
    state = env.reset()
    done = False
    while not done:
        action_prob, _ = model(tf.expand_dims(tf.convert_to_tensor(state), 0))
        # print("AP IS::::", action_prob)
        # action = tf.expand_dims(tf.argmax(tf.squeeze(action_prob)), 0)
        action = np.argmax(np.squeeze(action_prob))
        # print("THEN ACTION IS ::::", action)
        state, reward, done, _ = env.step(action)
        total_reward += reward
    return total_reward


if __name__ == '__main__':
    # Make single env for testing model performance
    env = gym.make("CartPole-v0")
    num_inputs = len(env.observation_space.high)
    num_actions = env.action_space.n

    # Make multiple vectorized environments for synchronous training
    envs = [make_env() for i in range(NUM_ENVS)]
    envs = SubprocVecEnv(envs)

    # Initialize model
    model = actor_critic_network(num_inputs, 128, num_actions)
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    # Run training
    running_reward = 0
    for e in range(EPOCHS):
        run_episode(envs)

        if e % TEST_EPOCHS == 0:
            ep_rew = test_agent()
            running_reward = 0.05 * ep_rew + (1 - 0.05) * running_reward

            template = "running reward: {:.2f} | episode reward: {:.2f} at episode {}"
            print(template.format(running_reward, ep_rew, e))

        if running_reward > 195:
            print("Solved at episode {}!".format(e))
            break
