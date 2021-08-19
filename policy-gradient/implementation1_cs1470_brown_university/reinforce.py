import os
import gym
import numpy as np
import tensorflow as tf

# Killing optional CPU driver warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# DO NOT ALTER MODEL CLASS OUTSIDE OF TODOs. OTHERWISE, YOU RISK INCOMPATIBILITY
# WITH THE AUTOGRADER AND RECEIVING A LOWER GRADE.


class Reinforce(tf.keras.Model):
    def __init__(self, state_size, num_actions):
        """
        The Reinforce class that inherits from tf.keras.Model
        The forward pass calculates the policy for the agent given a batch of states.

        :param state_size: number of parameters that define the state. You don't necessarily have to use this,
                           but it can be used as the input size for your first dense layer.
        :param num_actions: number of actions in an environment
        """
        super(Reinforce, self).__init__()
        self.num_actions = num_actions

        # TODO: Define network parameters and optimizer
        self.state_size = state_size
        self.hidden_size = 32
        self.dense1 = tf.keras.layers.Dense(self.hidden_size, activation='relu')
        self.dense2 = tf.keras.layers.Dense(self.num_actions, activation='softmax')
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
        of each state in the episode. This is our "policy"
        """
        # TODO: implement this ~

        dense1_out = self.dense1(states)
        prbs = self.dense2(dense1_out)
        return prbs

    def loss(self, states, actions, discounted_rewards):
        """
        Computes the loss for the agent. Make sure to understand the handout clearly when implementing this.

        :param states: A batch of states of shape [episode_length, state_size]
        :param actions: History of actions taken at each timestep of the episode (represented as an [episode_length] array)
        :param discounted_rewards: Discounted rewards throughout a complete episode (represented as an [episode_length] array)
        :return: loss, a TensorFlow scalar
        """
        # TODO: implement this uWu
        # Hint: Use gather_nd to get the probability of each action that was actually taken in the episode.

        # Pass the states we visited back into our policy to get the probability
        # distribution over the actions
        # tf.squeeze removes dimensions of size 1 from the tensor
        prbs = tf.squeeze(self.call(states))  # [episode_length, num_actions]

        # We only actually want the probability of the action that we already took
        # so now we need to remove the other probabilities
        episode_len = actions.shape[0]  # scalar

        # Convert actions tf.Tensor([1,2,3]) to list of list [[1], [2], [3]]
        idx = np.reshape(actions, [episode_len, 1])
        pa = tf.gather_nd(prbs, idx, batch_dims=1) # [episode length] tensor

        # Calculate loss
        # If an action has a low prob and a high reward, or a high prob and a low
        # reward, the loss will be high
        # We take the log of each action-prob we took, then we multiple each one
        # by the discounted reward. Then we sum across the all of these to get
        # a scalar value
        loss = - tf.reduce_sum(tf.multiply(tf.math.log(pa), discounted_rewards))
        # print("pa:\n\n", pa)
        # print("log(pa):\n\n", tf.math.log(pa))
        # print("log(pa) * G:\n\n", tf.multiply(tf.math.log(pa), discounted_rewards))
        # print("loss:\n\n", loss)
        return loss
