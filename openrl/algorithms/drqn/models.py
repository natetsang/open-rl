import tensorflow as tf
from tensorflow.keras import layers


def drqn_discrete_network(state_dims: int,
                          num_actions: int,
                          num_timesteps: int,
                          num_hidden_fc_layers: int,
                          hidden_size: int) -> tf.keras.Model:
    """
    Creates a fully-connected deep RECURRENT Q-network for DISCRETE action spaces.

    Input:
    - state vector
    - number of timesteps to track in the recurrent layer
    Output:
    - Q-value of each discrete action

    :param state_dims: The number of state dimensions
    :param num_actions: The number of discrete actions
    :param num_timesteps: The number of timesteps to track for the recurrent layer
    :param num_hidden_fc_layers: The number of hidden layers
    :param hidden_size: The number of neurons for each layer (all layers are same)
    :return: tf.keras.Model
    """
    # Get state inputs and pass through one hidden layer
    inputs = layers.Input(shape=(num_timesteps, state_dims), name="input_state_layer")

    # RNN layer
    hidden_lstm = layers.LSTM(hidden_size, activation="tanh")(inputs)

    # Create shared hidden layers
    hidden = hidden_lstm
    for i in range(num_hidden_fc_layers):
        hidden = layers.Dense(hidden_size, activation="relu", name=f"hidden_layer{i}")(hidden)
    outputs = layers.Dense(num_actions, name="output_layer")(hidden)

    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model


def dueling_dqn_fc_discrete_network(state_dims: int,
                                    num_actions: int,
                                    num_hidden_layers: int,
                                    hidden_size: int) -> tf.keras.Model:
    """
    Creates fully connected DUELING deep Q-network for use in DISCRETE action spaces.

    Input:
    - state vector
    Output:
    - Outputs values V(s) and the advantages A(s,a) for each action. Then uses these values
    to compute and ultimately output Q-values

    :param state_dims: The number of state dimensions
    :param num_actions: The number of discrete actions
    :param num_hidden_layers: The number of hidden layers in the network
    :param hidden_size: The number of neurons for each layer (all layers are same).
    :return: tf.keras.Model
    """
    # Get state inputs and pass through one hidden layer
    inputs = layers.Input(shape=(state_dims,), name="input_state_layer")

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
