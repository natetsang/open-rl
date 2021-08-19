"""
My first attempt at A3C
"""

import gym
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import multiprocessing as mp

# Set up
GAMMA = 0.99
LEARNING_RATE = 0.001
ENTROPY_WEIGHT = 0.01


###############################################################################

class Worker:
    def __init__(self, env):
        self.env = env
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.local_model = self.actor_critic_network(self.state_size, 128, self.action_size)
        self.local_optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

        # self.global_model = master_optimizer
        # self.global_optimizer = master_optimizer

    def actor_critic_network(self, num_inputs, num_hidden, num_actions):
        inputs = layers.Input(shape=(num_inputs,), name="Input_layer")
        common = layers.Dense(num_hidden, activation="relu", name="Common_hidden_layer")(inputs)
        action = layers.Dense(num_actions, activation="softmax", name="Action_output_layer")(common)
        critic = layers.Dense(1, name="Critic_output_layer")(common)
        model = tf.keras.Model(inputs=inputs, outputs=[action, critic])
        return model

    def calculate_advantages(self, reward_history, state_history, next_state, reached_done, gamma=GAMMA):
        # Get bootstrapped value for the n+1 step
        _, bootstrapped_value = self.local_model(tf.expand_dims(tf.convert_to_tensor(next_state, dtype=tf.float32), 0))
        bootstrapped_value = tf.squeeze(bootstrapped_value)

        # Discount rewards
        discounted_rewards = []
        total_ret = 0.0 if reached_done else bootstrapped_value
        for r in reward_history[::-1]:
            total_ret = r + gamma * total_ret
            discounted_rewards.append(total_ret)
        discounted_rewards.reverse()
        discounted_rewards = tf.convert_to_tensor(discounted_rewards)

        # Calculate critic values V(s) for each state in the trajectory
        _, critic_values = self.local_model(tf.convert_to_tensor(np.vstack(state_history), dtype=tf.float32))

        # Calculate advantages
        advantages = discounted_rewards - tf.squeeze(critic_values)
        return advantages

    def run(self):
        n_steps = 1000
        ep_rewards = 0
        state = self.env.reset()
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
                    action_prob, _ = self.local_model(state)
                    action = np.random.choice(self.action_size, p=np.squeeze(action_prob))
                    state, reward, done, _ = self.env.step(action)

                    # Some bookkeeping
                    ep_rewards += reward
                    reward_trajectory.append(reward)
                    prob_trajectory.append(action_prob)
                    action_prob_trajectory.append(action_prob[0, action])

                # Calculate entropies
                prob_trajectory = tf.convert_to_tensor(np.vstack(prob_trajectory), dtype=tf.float32)
                entropies = tf.reduce_sum(prob_trajectory * tf.math.log(prob_trajectory + 1e-8), axis=1)

                # Calculate advantages
                A = self.calculate_advantages(reward_trajectory, state_trajectory, state, done)

                # Calculate losses
                actor_loss = -tf.math.log(tf.convert_to_tensor(action_prob_trajectory)) * tf.stop_gradient(A)
                actor_loss += ENTROPY_WEIGHT * entropies
                critic_loss = tf.square(A)
                total_loss = tf.reduce_mean(actor_loss + 0.5 * critic_loss)

            # Calculate local gradient
            grads = tape.gradient(total_loss, self.local_model.trainable_variables)
            self.local_optimizer.apply_gradients(zip(grads, self.local_model.trainable_variables))
        return ep_rewards


###############################################################################
class MasterAgent:
    def __init__(self, env):
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.model = self.actor_critic_network(self.state_size , 128, self.action_size)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)

    def actor_critic_network(self, num_inputs, num_hidden, num_actions):
        inputs = layers.Input(shape=(num_inputs,), name="Input_layer")
        common = layers.Dense(num_hidden, activation="relu", name="Common_hidden_layer")(inputs)
        action = layers.Dense(num_actions, activation="softmax", name="Action_output_layer")(common)
        critic = layers.Dense(1, name="Critic_output_layer")(common)
        model = tf.keras.Model(inputs=inputs, outputs=[action, critic])
        return model


def master_agent(queue):
    master = MasterAgent(gym.make("CartPole-v0"))

def worker_agent(queue):
    w = Worker(gym.make("CartPole-v0"))

def main():
    queue = mp.Queue()

    p1 = mp.Process(target=master_agent, args=(queue,))
    p2 = mp.Process(target=worker_agent, args=(queue,))
    p1.start()
    print("Starting...p1")
    p2.start()
    print("Starting...p2")
    p1.join()
    p2.join()
    print("done")

if __name__ == '__main__':
    main()
