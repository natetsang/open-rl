import tensorflow as tf
from tensorflow.keras import layers


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


def dueling_dqn_fc_discrete_network(num_inputs: int,
                                    num_actions: int,
                                    num_hidden_layers: int,
                                    hidden_size: int) -> tf.keras.Model:
    """
    Creates DUELING deep Q-network for use in discrete-action space
    This model is fully connected and takes in both the state and outputs one Q-value per action
    as input. It outputs V(s) and the advantages A(s,a) for each action! Then, we use these values
    to compute and output the Q-values.

    :param num_inputs: The dimensionality of the observed state
    :param num_actions: The dimensionality of the action space
    :param num_hidden_layers: The number of hidden layers in the network
    :param hidden_size: The number of neurons for each layer.
        Note that all layers have the same hidden_size.
    :return: tf.keras.Model!
    """
    # Get state inputs and pass through one hidden layer
    inputs = layers.Input(shape=(num_inputs,), name="input_state_layer")

    # Create shared hidden layers
    hidden = inputs
    for i in range(num_hidden_layers):
        hidden = layers.Dense(hidden_size, activation="relu", name=f"hidden_layer{i}")(hidden)

    # Output both V(s) and A(s,a)
    values = layers.Dense(1, name="output_value_layer")(hidden)
    advantages = layers.Dense(num_actions, name="output_advantage_layer")(hidden)

    # Combine to calculate Q-values. Notice that we subtract out the average advantage
    q_values = values + (advantages - tf.reduce_mean(advantages))

    model = tf.keras.Model(inputs=inputs, outputs=q_values)
    return model
