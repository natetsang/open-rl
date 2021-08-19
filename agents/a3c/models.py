import tensorflow as tf
from tensorflow.keras import layers


def actor_critic_fc_discrete_network(num_inputs, num_hidden_layers, hidden_size, num_actions):
    inputs = layers.Input(shape=(num_inputs,), name="input_layer")

    # Create shared hidden layers
    hidden = inputs
    for i in range(num_hidden_layers):
        hidden = layers.Dense(hidden_size, activation="relu", name=f"hidden_layer{i}")(hidden)

    # Create output layers
    action = layers.Dense(num_actions, activation="softmax", name="action_output_layer")(hidden)
    critic = layers.Dense(1, name="critic_output_layer")(hidden)
    model = tf.keras.Model(inputs=inputs, outputs=[action, critic])
    return model
