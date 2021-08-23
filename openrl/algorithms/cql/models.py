import tensorflow as tf
from tensorflow.keras import layers


def dqn_fc_discrete_network(state_dims: int,
                            num_actions: int,
                            num_hidden_layers: int,
                            hidden_size: int) -> tf.keras.Model:
    """
    Creates a fully-connected deep Q-network for DISCRETE action spaces.

    Input:
    - state vector
    Output:
    - Q-value of each discrete action

    :param state_dims: The number of state dimensions
    :param num_actions: The number of discrete actions
    :param num_hidden_layers: The number of hidden layers
    :param hidden_size: The number of neurons in each hidden layer (all layers are same)
    :return: tf.keras.Model
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

