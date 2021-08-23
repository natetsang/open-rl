import tensorflow as tf
from tensorflow.keras import layers
from typing import Union


def td3_actor_fc_continuous_network(state_dims: int,
                                    action_dims: int,
                                    env_action_lb: Union[int, float],
                                    env_action_ub: Union[int, float]) -> tf.keras.Model:
    """
    Creates fully-connected TD3 actor model for CONTINUOUS action spaces
    using the exact model architecture as described in the original paper: https://arxiv.org/pdf/1802.09477.pdf

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


def td3_critic_fc_continuous_network(state_dims: int,
                                     action_dims: int) -> tf.keras.Model:
    """
    Creates fully-connected TD3 Critic model for CONTINUOUS action spaces
    using the exact model architecture as described in the original paper: https://arxiv.org/pdf/1802.09477.pdf

    Input:
    - state vector
    - action vector
    Output:
    - value of being in the current state and taking a particular action

    :param state_dims: The number of state dimensions
    :param action_dims: The number of action dimensions
    :return: tf.keras.Model
    """
    # Get state inputs and pass through one hidden layer
    state_inputs = layers.Input(shape=(state_dims,), name="input_state_layer")
    action_inputs = layers.Input(shape=(action_dims,), name="input_action_layer")
    inputs_concat = layers.Concatenate(name="concatenated_layer")([state_inputs, action_inputs])

    # Get concatenated inputs and pass through two hidden layers
    hidden1 = layers.Dense(400, activation="relu", name="hidden1_layer")(inputs_concat)
    hidden2 = layers.Dense(300, activation="relu", name="hidden2_layer")(hidden1)
    output = layers.Dense(1, name="output_layer")(hidden2)

    model = tf.keras.Model(inputs=[state_inputs, action_inputs], outputs=output)
    return model
