import tensorflow as tf
import numpy as np


def calculate_advantages(model, reward_history, state_history, next_state, reached_done, gamma=0.99):
    # Get bootstrapped value for the n+1 step
    _, bootstrapped_value = model(tf.expand_dims(tf.convert_to_tensor(next_state, dtype=tf.float32), 0))
    bootstrapped_value = tf.squeeze(bootstrapped_value)

    # Discount rewards
    discounted_rewards = []
    total_ret = 0.0 if reached_done else bootstrapped_value
    for r in reward_history[::-1]:
        total_ret = r + gamma * total_ret
        discounted_rewards.append(total_ret)
    discounted_rewards.reverse()
    discounted_rewards = tf.convert_to_tensor(discounted_rewards)

    # Normalize discounted rewards if n-step > 1
    if len(discounted_rewards) > 1:
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= (np.std(discounted_rewards) + 1e-8)

    # Calculate critic values V(s) for each state in the trajectory
    _, critic_values = model(tf.convert_to_tensor(np.vstack(state_history), dtype=tf.float32))

    # Calculate advantages
    advantages = discounted_rewards - tf.squeeze(critic_values)
    return advantages
