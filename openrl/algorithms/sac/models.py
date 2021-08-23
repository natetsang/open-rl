from typing import Union
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_probability as tfp
tfd = tfp.distributions


def sac_actor_fc_continuous_network(state_dims: int,
                                    action_dims: int,
                                    env_action_lb: Union[int, float],
                                    env_action_ub: Union[int, float],
                                    log_std_min: Union[int, float],
                                    log_std_max: Union[int, float],
                                    num_hidden_layers: int = 2,
                                    hidden_size: int = 256,
                                    ) -> tf.keras.Model:
    """
    Creates fully-connected SAC Actor model for CONTINUOUS action spaces
    using the exact model architecture as described in the original paper: https://arxiv.org/pdf/1802.09477.pdf

    Input:
    - state vector
    Output:
    - action to take (stochastic via reparameterization trick)

    :param state_dims: The number of state dimensions
    :param action_dims: The number of action dimensions
    :param env_action_lb: The environment's action upper bound
    :param env_action_ub: The environment's action upper bound
    :param log_std_min: The minimum permitted log of the standard deviation
    :param log_std_max: The maximum permitted log of the standard deviation
    :param num_hidden_layers: The number of hidden layers
    :param hidden_size: The number of neurons in each hidden layer (all layers are same)
    :return: tf.keras.Model
    """
    inputs = layers.Input(shape=(state_dims,), name="input_layer")

    # Create shared hidden layers
    hidden = inputs
    for i in range(num_hidden_layers):
        hidden = layers.Dense(hidden_size, activation="relu", name=f"hidden_layer{i}")(hidden)

    # Output mean and log_std
    mu = layers.Dense(action_dims)(hidden)
    log_std = layers.Dense(action_dims)(hidden)
    log_std = tf.clip_by_value(log_std, log_std_min, log_std_max)

    # Create Normal distribution with outputs
    std = tf.math.exp(log_std)
    pi_dist = tfd.Normal(mu, std)

    # To obtain actions, we sample from the distribution
    # We use the reparameterization trick here
    action = pi_dist.sample()

    # Get the log probability of that action w.r.t the distribution
    logp_pi = tf.reduce_sum(pi_dist.log_prob(action), axis=1)

    # NOTE: The correction formula is a little bit magic. To get an understanding
    # of where it comes from, check out the original SAC paper (arXiv 1801.01290)
    # and look in appendix C. This is a more numerically-stable equivalent to Eq 21.
    logp_pi -= tf.reduce_sum(2. * (tf.math.log(2.) - action - tf.math.softplus(-2. * action)), axis=1)

    # Squash the Gaussian policy
    action_squashed = tf.math.tanh(action)

    # Now either change output by multiplying by max_action,
    # action_squashed = action_squashed * self.env_action_ub

    # OR scale outputs to be within environment action space bounds
    resize_fn = lambda x: (((x + 1) * (env_action_ub - env_action_lb)) / 2) + env_action_lb
    action_squashed = resize_fn(action_squashed)

    model = tf.keras.Model(inputs=inputs, outputs=[action_squashed, logp_pi])
    return model


def sac_critic_fc_continuous_network(state_dims: int,
                                     action_dims: int,
                                     num_hidden_layers: int = 2,
                                     hidden_size: int = 256) -> tf.keras.Model:
    """
    Creates fully-connected SAC Critic model for CONTINUOUS action spaces.

    Input:
    - state vector
    - action vector
    Output:
    - value of being in the current state and taking a particular action

    :param state_dims: The number of state dimensions
    :param action_dims: The number of action dimensions
    :param num_hidden_layers: The number of hidden layers
    :param hidden_size: The number of neurons in each hidden layer (all layers are same)
    :return: tf.keras.Model
    """
    inputs_state = layers.Input(shape=(state_dims,), name="input_state_layer")
    inputs_action = layers.Input(shape=(action_dims,), name="input_action_layer")

    inputs_concat = layers.concatenate([inputs_state, inputs_action])

    # Create shared hidden layers
    hidden = inputs_concat
    for i in range(num_hidden_layers):
        hidden = layers.Dense(hidden_size, activation="relu", name=f"hidden_layer{i}")(hidden)

    # Create output layers
    critic = layers.Dense(1, name="critic_output_layer")(hidden)
    model = tf.keras.Model(inputs=[inputs_state, inputs_action], outputs=critic)
    return model
