import random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import matplotlib.pyplot as plt


class ContextBandit:
    def __init__(self, num_states=10, num_arms=10):
        self.n_arms = num_arms
        self.n_states = num_states
        self.bandit_matrix = np.random.rand(num_states, num_arms)
        self.reset()

    def reward(self, prob):
        """Given a probability, """
        reward = 0
        for i in range(self.n_arms):
            if random.random() < prob:
                reward += 1
        return reward

    def get_state(self):
        return self.state

    def reset(self):
        """Select a random discrete state"""
        self.state = np.random.randint(0, self.n_states)

    def get_reward(self, arm):
        return self.reward(self.bandit_matrix[self.get_state()][arm])

    def choose_arm(self, arm):
        rew = self.get_reward(arm)
        self.reset()
        return rew


def action_value_model(state_dims: int,
                       num_hidden_layers: int,
                       hidden_size: int,
                       num_actions: int) -> tf.keras.Model:
    inputs = layers.Input(shape=(state_dims,), name="input_layer")

    # Create shared hidden layers
    hidden = inputs
    for i in range(num_hidden_layers):
        hidden = layers.Dense(hidden_size, activation="relu", name=f"hidden_layer{i}")(hidden)

    # Create output layers
    output = layers.Dense(num_actions, name="output_layer")(hidden)

    model = tf.keras.Model(inputs=inputs, outputs=output)
    return model


def one_hot(N, pos, val=1):
    one_hot_vec = np.zeros(N)
    one_hot_vec[pos] = val
    return tf.convert_to_tensor([one_hot_vec])


def running_mean(x,N=50):
    c = x.shape[0] - N
    y = np.zeros(c)
    conv = np.ones(N)
    for i in range(c):
        y[i] = (x[i:i+N] @ conv)/N
    return y


if __name__ == "__main__":
    n_arms = 20
    n_states = 10
    env = ContextBandit(n_states, n_arms)
    model = action_value_model(state_dims=n_states, num_actions=n_arms, num_hidden_layers=2, hidden_size=100)
    opt = tf.keras.optimizers.Adam(learning_rate=0.01)
    rewards = []
    for i in range(5000):
        with tf.GradientTape() as tape:
            # Get new state, one hot encode, and pass it into the model to get predict rewards for each arm
            state = one_hot(n_states, env.get_state())
            pred_rewards = model(state)

            # Pass logits through softmax to get predicted probabilities for pulling each arm
            action_prob = tf.nn.softmax(pred_rewards)
            action = np.random.choice(n_arms, p=action_prob[0].numpy())

            # Observe reward from pulling an arm
            reward = env.choose_arm(action)
            rewards.append(reward)

            # Now we need to improve our action-value model
            # To do this, let's create a copy of our predicted rewards, and replace our r_pred for action a
            # with the reward we actually took.
            observed_rewards = pred_rewards.numpy()[0]
            observed_rewards[action] = reward
            observed_rewards = tf.expand_dims(observed_rewards, axis=0)

            # Compute loss between what we observed and what we predicted
            loss = tf.keras.losses.MSE(y_true=observed_rewards, y_pred=pred_rewards)
        grads = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(grads, model.trainable_variables))

    plt.plot(running_mean(np.array(rewards), N=500))
    plt.show()
