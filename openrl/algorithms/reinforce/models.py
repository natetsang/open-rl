import tensorflow as tf
from tensorflow.keras import layers


def policy_gradient_fc_discrete_network(num_inputs: int,
                                        num_hidden_layers: int,
                                        hidden_size: int,
                                        num_actions: int) -> tf.keras.Model:
    """
    Creates actor-critic model. This model is fully connected, takes in the state as input
    and outputs both the probability of taking each discrete action (i.e. actor) and the value (i.e. critic).

    :param num_inputs: The dimensionality of the observed state
    :param num_hidden_layers: The number of hidden layers in the fully-connected model
    :param hidden_size: The number of neurons in each hidden layer (note that all hidden layers have same number)
    :param num_actions: The dimensionality of the action space
    :return: tf.keras.Model!
    """
    inputs = layers.Input(shape=(num_inputs,), name="input_layer")

    # Create shared hidden layers
    hidden = inputs
    for i in range(num_hidden_layers):
        hidden = layers.Dense(hidden_size, activation="relu", name=f"hidden_layer{i}")(hidden)

    # Create output layers
    action = layers.Dense(num_actions, activation="softmax", name="action_output_layer")(hidden)
    model = tf.keras.Model(inputs=inputs, outputs=action)
    return model
