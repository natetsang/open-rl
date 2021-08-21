import tensorflow as tf
from tensorflow.keras import layers


def actor_fc_continuous_network(num_inputs: int,
                                num_hidden_layers: int,
                                hidden_size: int,
                                num_actions: int) -> tf.keras.Model:
    inputs = layers.Input(shape=(num_inputs,), name="input_layer")

    # Create shared hidden layers
    hidden = inputs
    for i in range(num_hidden_layers):
        hidden = layers.Dense(hidden_size, activation="relu", name=f"hidden_layer{i}")(hidden)

    mu = layers.Dense(num_actions, activation="tanh", name="mu")(hidden)
    mu = layers.Lambda(lambda x: x * 2.0)(mu)
    std = layers.Dense(num_actions, activation="softplus", name='std')(hidden)

    model = tf.keras.Model(inputs=inputs, outputs=[mu, std])
    return model


def critic_fc_network(num_inputs: int,
                      num_hidden_layers: int,
                      hidden_size: int) -> tf.keras.Model:
    inputs = layers.Input(shape=(num_inputs,), name="input_layer")

    # Create shared hidden layers
    hidden = inputs
    for i in range(num_hidden_layers):
        hidden = layers.Dense(hidden_size, activation="relu", name=f"hidden_layer{i}")(hidden)

    # Create output layers
    critic = layers.Dense(1, name="critic_output_layer")(hidden)

    model = tf.keras.Model(inputs=inputs, outputs=critic)
    return model
