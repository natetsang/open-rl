import tensorflow as tf
from tensorflow.keras import layers
from typing import Union


def td3_actor_fc_continuous_network(num_inputs: int,
                                    num_actions: int,
                                    env_action_lb: Union[int, float],
                                    env_action_ub: Union[int, float]) -> tf.keras.Model:
    """
    Creates TD3 actor model using the exact model architecture as described in
    the original paper: https://arxiv.org/pdf/1802.09477.pdf
    This model is fully connected, takes in the state as input
    and outputs the a deterministic action (i.e. actor).

    :param num_inputs: The dimensionality of the observed state
    :param num_actions: The dimensionality of the action space
    :param env_action_lb: The environment's action upper bound
    :param env_action_ub: The environment's action upper bound
    :return: tf.keras.Model!
    """
    inputs = layers.Input(shape=(num_inputs,), name="input_layer")

    # Create shared hidden layers
    hidden1 = layers.BatchNormalization()(inputs)
    hidden1 = layers.Dense(400, activation="relu", name="hidden1_layer")(hidden1)
    hidden2 = layers.BatchNormalization()(hidden1)
    hidden2 = layers.Dense(300, activation="relu", name="hidden2_layer")(hidden2)

    # Initialize weights of last layer [-3e-3, 3-e3]
    weight_init = tf.keras.initializers.RandomUniform(-0.003, 0.003)

    # Create output layers
    action = layers.BatchNormalization()(hidden2)
    action = layers.Dense(num_actions,
                          activation="tanh",
                          kernel_initializer=weight_init,
                          name="action_output_layer")(action)

    # Scale outputs to be within environment action space bounds
    resize_fn = lambda x: (((x + 1) * (env_action_ub - env_action_lb)) / 2) + env_action_lb
    action_scaled = resize_fn(action)

    model = tf.keras.Model(inputs=inputs, outputs=action_scaled)
    return model


def td3_critic_fc_continuous_network(num_inputs: int,
                                     num_actions: int) -> tf.keras.Model:
    """
    Creates TD3 critic model using the exact model architecture as described in
    the original paper: https://arxiv.org/pdf/1802.09477.pdf
    This model is fully connected andtakes in both the state and action
    as input. It outputs the value (i.e. critic).

    :param num_inputs: The dimensionality of the observed state
    :param num_actions: The dimensionality of the action space
    :return: tf.keras.Model!
    """
    # Get state inputs and pass through one hidden layer
    state_inputs = layers.Input(shape=(num_inputs,), name="input_state_layer")
    action_inputs = layers.Input(shape=(num_actions,), name="input_action_layer")
    inputs_concat = layers.Concatenate(name="concatenated_layer")([state_inputs, action_inputs])

    # Get concatenated inputs and pass through two hidden layers
    hidden1 = layers.Dense(400, activation="relu", name="hidden1_layer")(inputs_concat)
    hidden2 = layers.Dense(300, activation="relu", name="hidden2_layer")(hidden1)
    output = layers.Dense(1, name="output_layer")(hidden2)

    model = tf.keras.Model(inputs=[state_inputs, action_inputs], outputs=output)
    return model
