import os
import gym
import numpy as np
import tensorflow as tf

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# DO NOT ALTER MODEL CLASS OUTSIDE OF TODOs. OTHERWISE, YOU RISK INCOMPATIBILITY
# WITH THE AUTOGRADER AND RECEIVING A LOWER GRADE.


class ReinforceWithBaseline(tf.keras.Model):
    def __init__(self, state_size, num_actions):
        """
        The ReinforceWithBaseline class that inherits from tf.keras.Model.

        The forward pass calculates the policy for the agent given a batch of states. During training,
        ReinforceWithBaseLine estimates the value of each state to be used as a baseline to compare the policy's
        performance with.

        :param state_size: number of parameters that define the state. You don't necessarily have to use this,
                           but it can be used as the input size for your first dense layer.
        :param num_actions: number of actions in an environment
        """
        super(ReinforceWithBaseline, self).__init__()
        self.num_actions = num_actions

        # TODO: Define actor network parameters, critic network parameters, and optimizer
        self.state_size = state_size
        self.hidden_size = 32

        # Actor
        self.actor_dense1 = tf.keras.layers.Dense(self.hidden_size, activation='relu')
        self.actor_dense2 = tf.keras.layers.Dense(self.num_actions, activation='softmax')

        # Critic
        self.critic_dense1 = tf.keras.layers.Dense(self.hidden_size, activation='relu')
        self.critic_dense2 = tf.keras.layers.Dense(1)

        self.optimizer = tf.keras.optimizers.Adam(learning_rate=1e-3)

    @tf.function
    def call(self, states):
        """
        Performs the forward pass on a batch of states to generate the action probabilities.
        This returns a policy tensor of shape [episode_length, num_actions], where each row is a
        probability distribution over actions for each state.

        :param states: An [episode_length, state_size] dimensioned array
        representing the history of states of an episode
        :return: A [episode_length,num_actions] matrix representing the probability distribution over actions
        of each state in the episode
        """
        # TODO: implement this!
        dense1_out = self.actor_dense1(states)
        prbs = self.actor_dense2(dense1_out)
        return prbs

    def value_function(self, states):
        """
        Performs the forward pass on a batch of states to calculate the value function, to be used as the
        critic in the loss function.

        :param states: An [episode_length, state_size] dimensioned array representing the history of states
        of an episode
        :return: A [episode_length] matrix representing the value of each state
        """
        # TODO: implement this :D
        dense1_out = self.critic_dense1(states)
        values = self.critic_dense2(dense1_out)
        return values

    def loss(self, states, actions, discounted_rewards):
        """
        Computes the loss for the agent. Refer to the handout to see how this is done.

        Remember that the loss is similar to the loss as in reinforce.py, with one specific change.

        1) Instead of element-wise multiplying with discounted_rewards, you want to element-wise multiply with your advantage.
        Here, advantage is defined as discounted_rewards - state_values, where state_values is calculated by the critic network.

        2) In your actor loss, you must set advantage to be tf.stop_gradient(discounted_rewards - state_values).
        You may need to cast your (discounted_rewards - state_values) to tf.float32.
        tf.stop_gradient is used here to stop the loss calculated on the actor network from propagating back to the critic network.

        3) To calculate the loss for your critic network. Do this by calling the value_function on the states and then taking the sum of the squared advantage.

        The total loss should be equal to a weighted sum of the actor network loss and the critic network loss
        (we recommend the critic's loss to be weighted by 0.5).

        :param states: A batch of states of shape [episode_length, state_size]
        :param actions: History of actions taken at each timestep of the episode (represented as an [episode_length] array)
        :param discounted_rewards: Discounted rewards throughout a complete episode (represented as an [episode_length] array)
        :return: loss, a TensorFlow scalar
        """
        # TODO: implement this :)
        # Hint: use tf.gather_nd (https://www.tensorflow.org/api_docs/python/tf/gather_nd)
        # to get the probabilities of the actions taken by the model
        prbs = tf.squeeze(self.call(states))
        state_values = self.value_function(states)
        advantages = tf.stop_gradient(tf.cast(discounted_rewards - state_values, dtype=tf.float32))

        episode_len = actions.shape[0]
        idx = np.reshape(actions, [episode_len, 1])
        pa = tf.gather_nd(prbs, idx, batch_dims=1)
        actor_loss = - tf.reduce_sum(tf.multiply(tf.math.log(pa), advantages))

        critic_loss = tf.reduce_sum(tf.square((discounted_rewards - state_values)))

        loss = actor_loss + (critic_loss * 0.5)
        return loss
