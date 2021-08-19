import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_probability as tfp
tfd = tfp.distributions
tf.keras.backend.set_floatx('float32')


class Actor(tf.keras.Model):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.state_size = state_size
        self.action_size = action_size

        self.hidden1 = layers.Dense(256, activation="relu")
        self.hidden2 = layers.Dense(256, activation="relu")
        self.output_mu = layers.Dense(self.action_size)

        # Standard Deviation in Normal Distribution is a trainable variable and
        # is updated by the optimizer
        self.log_std = tf.Variable(name="LOG_STD", initial_value= -0.5 *
                                   np.ones(self.action_size, dtype= np.float32),
                                   trainable=True)

    def call(self, states):
        out = self.hidden1(states)
        out = self.hidden2(out)
        mu = self.output_mu(out)
        std = tf.math.exp(self.log_std)
        return tfd.Normal(mu, std)
