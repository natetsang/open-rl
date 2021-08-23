import tensorflow as tf
from tensorflow.keras import layers


def actor_critic_fc_discrete_network(state_dims: int,
                                     num_actions: int,
                                     num_hidden_layers: int,
                                     hidden_size: int) -> tf.keras.Model:
    """
    Creates a fully-connected Actor-Critic model for DISCRETE action spaces.

    Input:
    - state vector
    Output:
    - probability of taking each discrete action (Actor)
    - value of being in the current state (Critic)

    :param state_dims: The dimensionality of the observed state
    :param num_actions: The number of discrete actions
    :param num_hidden_layers: The number of hidden layers
    :param hidden_size: The number of neurons in each hidden layer (all layers are same)
    :return: tf.keras.Model
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


def actor_fc_discrete_network(state_dims: int,
                              num_actions: int,
                              num_hidden_layers: int,
                              hidden_size: int) -> tf.keras.Model:
    """
    Creates a fully connected Actor model for DISCRETE action spaces.

    Input:
    - state vector
    Output:
    - probability of taking each discrete action

    :param state_dims: The number of state dimensions
    :param num_actions: The number of discrete actions
    :param num_hidden_layers: The number of hidden layers
    :param hidden_size: The number of neurons in each hidden layer (all layers are same)
    :return: tf.keras.Model
    """
    inputs = layers.Input(shape=(state_dims,), name="input_layer")

    # Create shared hidden layers
    hidden = inputs
    for i in range(num_hidden_layers):
        hidden = layers.Dense(hidden_size, activation="relu", name=f"hidden_layer{i}")(hidden)

    # Create output layers
    action = layers.Dense(num_actions, activation="softmax", name="action_output_layer")(hidden)
    model = tf.keras.Model(inputs=inputs, outputs=action)
    return model


def critic_fc_network(state_dims: int,
                      num_hidden_layers: int,
                      hidden_size: int) -> tf.keras.Model:
    """
    Creates a fully connected Critic model.

    Input:
    - state vector
    Output:
    - value of being in the current state

    :param state_dims: The number of state dimensions
    :param num_hidden_layers: The number of hidden layers
    :param hidden_size: The number of neurons in each hidden layer (all layers are same)
    :return: tf.keras.Model
    """
    inputs = layers.Input(shape=(state_dims,), name="input_layer")

    # Create shared hidden layers
    hidden = inputs
    for i in range(num_hidden_layers):
        hidden = layers.Dense(hidden_size, activation="relu", name=f"hidden_layer{i}")(hidden)

    # Create output layers
    critic = layers.Dense(1, name="critic_output_layer")(hidden)
    model = tf.keras.Model(inputs=inputs, outputs=critic)
    return model


def actor_fc_continuous_network(state_dims: int,
                                action_dims: int,
                                num_hidden_layers: int,
                                hidden_size: int) -> tf.keras.Model:
    """
    Creates a fully-connected Actor model for CONTINUOUS action spaces.

    Input:
    - state vector
    Output:
    - mean action to take (for each action dim)
    - standard deviation of action to take (for each action dim)

    :param state_dims: The number of state dimensions
    :param action_dims: The number of action dimensions
    :param num_hidden_layers: The number of hidden layers
    :param hidden_size: The number of neurons in each hidden layer (all layers are same)
    :return: tf.keras.Model
    """
    inputs = layers.Input(shape=(state_dims,), name="input_layer")

    # Create shared hidden layers
    hidden = inputs
    for i in range(num_hidden_layers):
        hidden = layers.Dense(hidden_size, activation="relu", name=f"hidden_layer{i}")(hidden)

    mu = layers.Dense(action_dims, activation="tanh", name="mu")(hidden)
    mu = layers.Lambda(lambda x: x * 2.0)(mu)
    std = layers.Dense(action_dims, activation="softplus", name='std')(hidden)

    model = tf.keras.Model(inputs=inputs, outputs=[mu, std])
    return model
