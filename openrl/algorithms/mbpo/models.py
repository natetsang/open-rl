import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_probability as tfp
from .utils import normalize, unnormalize
import numpy as np
from typing import Union
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
    Creates SAC actor model using the exact model architecture as described in
    the original paper: https://arxiv.org/pdf/1802.09477.pdf
    This model is fully connected, takes in the state as input
    and outputs the a deterministic action (i.e. actor).

    :param state_dims: The dimensionality of the observed state
    :param action_dims: The dimensionality of the action space
    :param env_action_lb: The environment's action upper bound
    :param env_action_ub: The environment's action upper bound
    :param log_std_min: The minimum permitted log of the standard deviation
    :param log_std_max: The maximum permitted log of the standard deviation
    :param num_hidden_layers: The number of hidden layers in the fully-connected model
    :param hidden_size: The number of neurons in each hidden layer (note that all hidden layers have same number)
    :return: tf.keras.Model!
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


def critic_fc_network(state_dims: int,
                      action_dims: int,
                      num_hidden_layers: int = 2,
                      hidden_size: int = 256) -> tf.keras.Model:
    """
    Creates critic model. This model is fully connected, takes in the state AND action as inputs
    and outputs the value (i.e. critic).

    :param state_dims: The dimensionality of the observed state
    :param action_dims: The dimensionality of the action space
    :param num_hidden_layers: The number of hidden layers in the fully-connected model
    :param hidden_size: The number of neurons in each hidden layer (note that all hidden layers have same number)
    :return: tf.keras.Model!
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

####################################################################################################################
####################################################################################################################


def fc_network(state_dims: int,
               action_dims: int,
               num_hidden_layers: int,
               hidden_size: int) -> tf.keras.Model:
    """
    Input both normalized state and normalized action.
    Output the predicted normalized delta between the next state s' and the current state s.
    """
    # Get state inputs and pass through one hidden layer
    state_inputs = layers.Input(shape=(state_dims,), name="input_state_layer")
    action_inputs = layers.Input(shape=(action_dims,), name="input_action_layer")
    inputs_concat = layers.Concatenate(name="concatenated_layer")([state_inputs, action_inputs])

    # Create shared hidden layers
    hidden = inputs_concat
    for i in range(num_hidden_layers):
        hidden = layers.Dense(hidden_size, activation="relu", name=f"hidden_layer{i}")(hidden)
    next_state_outputs = layers.Dense(state_dims, name="output_layer")(hidden)

    model = tf.keras.Model(inputs=[state_inputs, action_inputs], outputs=next_state_outputs)
    return model


class FFModel:
    def __init__(self, ac_dim, ob_dim, n_layers, hidden_size, learning_rate=0.001):
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.delta_network = fc_network(state_dims=ob_dim,
                                        action_dims=ac_dim,
                                        num_hidden_layers=n_layers,
                                        hidden_size=hidden_size)
        self.trainable_variables = self.delta_network.trainable_variables

    def _forward_pass(self, obs: np.ndarray, acs: np.ndarray, data_statistics: dict) -> np.ndarray:
        """
        Normalize the observations and actions and pass it through our NN to get our prediction.
        Recall that our NN outputs the normalized delta between s' and s'.
        """
        obs_normalized = normalize(obs, data_statistics['obs_mean'], data_statistics['obs_std'])
        acs_normalized = normalize(acs, data_statistics['acs_mean'], data_statistics['acs_std'])
        delta_pred_normalized = self.delta_network([obs_normalized, acs_normalized])
        return delta_pred_normalized

    def get_prediction(self, obs: np.ndarray, acs: np.ndarray, data_statistics: dict) -> np.ndarray:
        """
        Predict the next states!
        :param obs: numpy array of unnormalized observations (s_t)
        :param acs: numpy array of unnormalized actions (a_t)
        :param data_statistics: A dictionary with the following keys (each with
        a numpy array as the value):
             - 'obs_mean'
             - 'obs_std'
             - 'acs_mean'
             - 'acs_std'
             - 'delta_mean'
             - 'delta_std'
        :return: a numpy array of the predicted next-states (s_t+1)
        """
        delta_pred_normalized = self._forward_pass(obs, acs, data_statistics)
        delta_pred_unnormalized = unnormalize(delta_pred_normalized,
                                              data_statistics['delta_mean'],
                                              data_statistics['delta_std'])
        pred_next_state = obs + delta_pred_unnormalized
        return pred_next_state

    def loss(self, obs: np.ndarray, acs: np.ndarray, next_obs: np.ndarray, data_statistics: dict) -> float:
        """
        Calculate and return the L2 model loss.
        """
        delta_pred_normalized = self._forward_pass(obs, acs, data_statistics)
        targets = normalize(next_obs - obs,
                            data_statistics['delta_mean'],
                            data_statistics['delta_std'])
        return tf.reduce_mean(tf.square(targets - delta_pred_normalized))


