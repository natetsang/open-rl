import tensorflow as tf
from tensorflow.keras import layers
from .utils import normalize, unnormalize
import numpy as np


def fc_discrete_network(state_dims: int,
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


def fc_reward_network(state_dims: int,
                      action_dims: int,
                      num_hidden_layers: int,
                      hidden_size: int) -> tf.keras.Model:

    # Get state inputs and pass through one hidden layer
    state_inputs = layers.Input(shape=(state_dims,), name="input_state_layer")
    action_inputs = layers.Input(shape=(action_dims,), name="input_action_layer")
    inputs_concat = layers.Concatenate(name="concatenated_layer")([state_inputs, action_inputs])

    # Create shared hidden layers
    hidden = inputs_concat
    for i in range(num_hidden_layers):
        hidden = layers.Dense(hidden_size, activation="relu", name=f"hidden_layer{i}")(hidden)
    reward_output = layers.Dense(1, name="output_layer")(hidden)

    model = tf.keras.Model(inputs=[state_inputs, action_inputs], outputs=reward_output)
    return model


def actor_critic_fc_discrete_network(state_dims: int,
                                     num_actions: int,
                                     num_hidden_layers: int,
                                     hidden_size: int) -> tf.keras.Model:
    """
    Creates actor-critic model. This model is fully connected, takes in the state as input
    and outputs both the probability of taking each discrete action (i.e. actor) and the value (i.e. critic).

    :param state_dims: The dimensionality of the observed state
    :param num_actions: The dimensionality of the action space
    :param num_hidden_layers: The number of hidden layers in the fully-connected model
    :param hidden_size: The number of neurons in each hidden layer (note that all hidden layers have same number)
    :return: tf.keras.Model!
    """
    inputs = layers.Input(shape=(state_dims,), name="input_layer")

    # Create shared hidden layers
    hidden = inputs
    for i in range(num_hidden_layers):
        hidden = layers.Dense(hidden_size, activation="relu", name=f"hidden_layer{i}")(hidden)

    # Create output layers
    action = layers.Dense(num_actions, activation="softmax", name="action_output_layer")(hidden)
    critic = layers.Dense(1, name="critic_output_layer")(hidden)
    model = tf.keras.Model(inputs=inputs, outputs=[action, critic])
    return model


class FFModel:
    def __init__(self, ac_dim, ob_dim, n_layers, hidden_size, learning_rate=0.001):
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        self.delta_network = fc_discrete_network(state_dims=ob_dim,
                                                 action_dims=ac_dim,
                                                 num_hidden_layers=n_layers,
                                                 hidden_size=hidden_size)
        self.trainable_variables = self.delta_network.trainable_variables

    def forward_pass(self, obs: np.ndarray, acs: np.ndarray, data_statistics: dict) -> np.ndarray:
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
        delta_pred_normalized = self.forward_pass(obs, acs, data_statistics)
        delta_pred_unnormalized = unnormalize(delta_pred_normalized,
                                              data_statistics['delta_mean'],
                                              data_statistics['delta_std'])
        pred_next_state = obs + delta_pred_unnormalized
        return pred_next_state

    def loss(self, obs: np.ndarray, acs: np.ndarray, next_obs: np.ndarray, data_statistics: dict) -> float:
        """
        Calculate and return the L2 model loss.
        """
        delta_pred_normalized = self.forward_pass(obs, acs, data_statistics)
        targets = normalize(next_obs - obs,
                            data_statistics['delta_mean'],
                            data_statistics['delta_std'])
        return tf.reduce_mean(tf.square(targets - delta_pred_normalized))


def dqn_fc_discrete_network(state_dims: int,
                            num_actions: int,
                            num_hidden_layers: int,
                            hidden_size: int) -> tf.keras.Model:
    """
    Creates deep Q-network for use in discrete-action space
    This model is fully connected and takes in both the state and outputs one Q-value per action
    as input. It outputs the Q-value.

    :param state_dims: The dimensionality of the observed state
    :param num_actions: The dimensionality of the action space
    :param num_hidden_layers: The number of hidden layers in the network
    :param hidden_size: The number of neurons for each layer. Note that all layers have
        the same hidden_size.
    :return: tf.keras.Model!
    """
    # Get state inputs and pass through one hidden layer
    inputs = layers.Input(shape=(state_dims,), name="input_state_layer")

    # Create shared hidden layers
    hidden = inputs
    for i in range(num_hidden_layers):
        hidden = layers.Dense(hidden_size, activation="relu", name=f"hidden_layer{i}")(hidden)
    outputs = layers.Dense(num_actions, name="output_layer")(hidden)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model
