import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_probability as tfp
from .utils import normalize, unnormalize
import numpy as np
from typing import Union, Tuple
tfd = tfp.distributions


def dqn_fc_discrete_network(num_inputs: int,
                            num_actions: int,
                            num_hidden_layers: int,
                            hidden_size: int) -> tf.keras.Model:
    """
    Creates deep Q-network for use in discrete-action space
    This model is fully connected and takes in both the state and outputs one Q-value per action
    as input. It outputs the Q-value.

    :param num_inputs: The dimensionality of the observed state
    :param num_actions: The dimensionality of the action space
    :param num_hidden_layers: The number of hidden layers in the network
    :param hidden_size: The number of neurons for each layer. Note that all layers have
        the same hidden_size.
    :return: tf.keras.Model!
    """
    # Get state inputs and pass through one hidden layer
    inputs = layers.Input(shape=(num_inputs,), name="input_state_layer")

    # Create shared hidden layers
    hidden = inputs
    for i in range(num_hidden_layers):
        hidden = layers.Dense(hidden_size, activation="relu", name=f"hidden_layer{i}")(hidden)
    outputs = layers.Dense(num_actions, name="output_layer")(hidden)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def sac_actor_fc_continuous_network(num_inputs: int,
                                    num_actions: int,
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

    :param num_inputs: The dimensionality of the observed state
    :param num_actions: The dimensionality of the action space
    :param env_action_lb: The environment's action upper bound
    :param env_action_ub: The environment's action upper bound
    :param log_std_min: The minimum permitted log of the standard deviation
    :param log_std_max: The maximum permitted log of the standard deviation
    :param num_hidden_layers: The number of hidden layers in the fully-connected model
    :param hidden_size: The number of neurons in each hidden layer (note that all hidden layers have same number)
    :return: tf.keras.Model!
    """
    inputs = layers.Input(shape=(num_inputs,), name="input_layer")

    # Create shared hidden layers
    hidden = inputs
    for i in range(num_hidden_layers):
        hidden = layers.Dense(hidden_size, activation="relu", name=f"hidden_layer{i}")(hidden)

    # Output mean and log_std
    mu = layers.Dense(num_actions)(hidden)
    log_std = layers.Dense(num_actions)(hidden)
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


def critic_fc_network(num_inputs: int,
                      num_actions: int,
                      num_hidden_layers: int = 2,
                      hidden_size: int = 256) -> tf.keras.Model:
    """
    Creates critic model. This model is fully connected, takes in the state AND action as inputs
    and outputs the value (i.e. critic).

    :param num_inputs: The dimensionality of the observed state
    :param num_actions: The dimensionality of the action space
    :param num_hidden_layers: The number of hidden layers in the fully-connected model
    :param hidden_size: The number of neurons in each hidden layer (note that all hidden layers have same number)
    :return: tf.keras.Model!
    """
    inputs_state = layers.Input(shape=(num_inputs,), name="input_state_layer")
    inputs_action = layers.Input(shape=(num_actions,), name="input_action_layer")

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


def fc_discrete_network(num_inputs: int,
                        num_actions: int,
                        num_hidden_layers: int,
                        hidden_size: int) -> tf.keras.Model:
    """
    Input both normalized state and normalized action.
    Output the predicted normalized delta between the next state s' and the current state s.
    """
    # Get state inputs and pass through one hidden layer
    state_inputs = layers.Input(shape=(num_inputs,), name="input_state_layer")
    action_inputs = layers.Input(shape=(num_actions,), name="input_action_layer")
    inputs_concat = layers.Concatenate(name="concatenated_layer")([state_inputs, action_inputs])

    # Create shared hidden layers
    hidden = inputs_concat
    for i in range(num_hidden_layers):
        hidden = layers.Dense(hidden_size, activation="relu", name=f"hidden_layer{i}")(hidden)

    output_mus = layers.Dense(num_inputs, name="output_mus")(hidden)
    output_stds = layers.Dense(num_inputs, activation="softplus", name="output_stds")(hidden)

    model = tf.keras.Model(inputs=[state_inputs, action_inputs], outputs=[output_mus, output_stds])
    return model


class FFModel:
    def __init__(self, ac_dim, ob_dim, n_layers, hidden_size, learning_rate=0.001):
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.delta_network = fc_discrete_network(num_inputs=ob_dim,
                                                 num_actions=ac_dim,
                                                 num_hidden_layers=n_layers,
                                                 hidden_size=hidden_size)
        self.trainable_variables = self.delta_network.trainable_variables

    def forward_pass(self, obs: np.ndarray, acs: np.ndarray, data_statistics: dict) -> Tuple[np.ndarray, np.ndarray]:
        """
        Normalize the observations and actions and pass it through our NN to get our prediction.
        Recall that our NN outputs the normalized delta between s' and s'.
        """
        obs_normalized = normalize(obs, data_statistics['obs_mean'], data_statistics['obs_std'])
        acs_normalized = normalize(acs, data_statistics['acs_mean'], data_statistics['acs_std'])
        delta_pred_normalized_mus, delta_pred_normalized_stds = self.delta_network([obs_normalized, acs_normalized])
        return delta_pred_normalized_mus, delta_pred_normalized_stds

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
        delta_pred_normalized_mus, delta_pred_normalized_stds = self.forward_pass(obs, acs, data_statistics)
        delta_pred_normalized_dist = tfd.MultivariateNormalDiag(loc=delta_pred_normalized_mus, scale_diag=delta_pred_normalized_stds)
        delta_pred_normalized = delta_pred_normalized_dist.sample()
        delta_pred_unnormalized = unnormalize(delta_pred_normalized,
                                              data_statistics['delta_mean'],
                                              data_statistics['delta_std'])
        pred_next_state = obs + delta_pred_unnormalized
        return pred_next_state

    def loss(self, obs: np.ndarray, acs: np.ndarray, next_obs: np.ndarray, data_statistics: dict) -> float:
        """
        Calculate and return the negative log-likelihood model loss.
        """
        delta_pred_normalized_mus, delta_pred_normalized_stds = self.forward_pass(obs, acs, data_statistics)
        delta_pred_normalized_dist = tfd.MultivariateNormalDiag(loc=delta_pred_normalized_mus, scale_diag=delta_pred_normalized_stds)

        targets = normalize(next_obs - obs,
                            data_statistics['delta_mean'],
                            data_statistics['delta_std'])
        return -delta_pred_normalized_dist.log_prob(targets)

