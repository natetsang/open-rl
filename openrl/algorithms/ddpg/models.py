import tensorflow as tf
from tensorflow.keras import layers
from typing import Union


def ddpg_actor_fc_continuous_network(state_dims: int,
                                     action_dims: int,
                                     env_action_lb: Union[int, float],
                                     env_action_ub: Union[int, float]) -> tf.keras.Model:
    """
    Creates fully-connected DDPG Actor model for CONTINUOUS action spaces
    using the exact model architecture as described in the original paper: https://arxiv.org/abs/1509.02971

    Input:
    - state vector
    Output:
    - action to take (deterministic)

    :param state_dims: The number of state dimensions
    :param action_dims: The number of action dimensions
    :param env_action_lb: The environment's action upper bound
    :param env_action_ub: The environment's action upper bound
    :return: tf.keras.Model
    """
    inputs = layers.Input(shape=(state_dims,), name="input_layer")

    # Create shared hidden layers
    hidden1 = layers.BatchNormalization()(inputs)
    hidden1 = layers.Dense(400, activation="relu", name="hidden1_layer")(hidden1)
    hidden2 = layers.BatchNormalization()(hidden1)
    hidden2 = layers.Dense(300, activation="relu", name="hidden2_layer")(hidden2)

    # Initialize weights of last layer [-3e-3, 3-e3]
    weight_init = tf.keras.initializers.RandomUniform(-0.003, 0.003)

    # Create output layers
    action = layers.BatchNormalization()(hidden2)
    action = layers.Dense(action_dims,
                          activation="tanh",
                          kernel_initializer=weight_init,
                          name="action_output_layer")(action)

    # Scale outputs to be within environment action space bounds
    resize_fn = lambda x: (((x + 1) * (env_action_ub - env_action_lb)) / 2) + env_action_lb
    action_scaled = resize_fn(action)

    model = tf.keras.Model(inputs=inputs, outputs=action_scaled)
    return model


def ddpg_critic_fc_continuous_network(state_dims: int,
                                      action_dims: int,
                                      l2_reg_factor: float = 0.01) -> tf.keras.Model:
    """
    Creates fully-connected DDPG Critic model for CONTINUOUS action spaces
    using the exact model architecture as described in the original paper: https://arxiv.org/abs/1509.02971

    Input:
    - state vector
    Output:
    - value of being in the current state

    :param state_dims: The number of state dimensions
    :param action_dims: The number of action dimensions
    :param l2_reg_factor: L2 regularization factor (i.e. penalty)
    :return: tf.keras.Model
    """
    # Get state inputs and pass through one hidden layer
    state_inputs = layers.Input(shape=(state_dims,), name="input_state_layer")
    hidden_state = layers.BatchNormalization()(state_inputs)
    hidden_state = layers.Dense(400,
                                activation="relu",
                                kernel_regularizer=tf.keras.regularizers.l2(l2_reg_factor),
                                name="hidden_state_layer")(hidden_state)

    # Get action inputs and pass through one hidden layer
    action_inputs = layers.Input(shape=(action_dims,), name="input_action_layer")
    hidden_action = layers.BatchNormalization()(action_inputs)
    hidden_action = layers.Dense(400,
                                 activation="relu",
                                 kernel_regularizer=tf.keras.regularizers.l2(l2_reg_factor),
                                 name="hidden_action_layer")(hidden_action)

    # Concatenate two layers and pass through another hidden layer
    hidden2 = layers.Concatenate(name="concatenated_layer")([hidden_state, hidden_action])
    hidden2 = layers.BatchNormalization()(hidden2)
    hidden2 = layers.Dense(300,
                           activation="relu",
                           kernel_regularizer=tf.keras.regularizers.l2(l2_reg_factor),
                           name="hidden2")(hidden2)

    # Initialize weights of last layer [-3e-3, 3-e3]
    weight_init = tf.keras.initializers.RandomUniform(-0.003, 0.003)

    output = layers.BatchNormalization()(hidden2)
    output = layers.Dense(1,
                          kernel_initializer=weight_init,
                          kernel_regularizer=tf.keras.regularizers.l2(l2_reg_factor),
                          name="output_layer")(output)

    model = tf.keras.Model(inputs=[state_inputs, action_inputs], outputs=output)
    return model
