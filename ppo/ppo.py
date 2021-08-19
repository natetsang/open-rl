"""
My first attempt at PPO!
"""
import os, logging
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.CRITICAL)
logging.getLogger("tensorflow_hub").setLevel(logging.CRITICAL)

import gym
import numpy as np
import scipy.signal
import tensorflow as tf
from tensorflow.keras import layers
import tensorflow_probability as tfp
tfd = tfp.distributions

from ppo_buffer import PPOBuffer
from ppo_actor import Actor

tf.keras.backend.set_floatx('float32')

# Set up
GAMMA = 0.99
LAMBDA = 0.95
CLIP_PARAM = 0.2
ACTOR_LEARNING_RATE = 1e-3
CRITIC_LEARNING_RATE = 1e-3
ENTROPY_WEIGHT = 0.001

NUM_STEPS_PER_ROLLOUT = 2048  #2048  #2048  or 2
TARGET_KL = 0.05
MINIBATCH_SIZE = 64 # 64 # 64 or 1 or 2
NUM_EPISODES = 101 #  51  # 10+ or 1

# This represents one full pass through the buffer
k_train_iters = NUM_STEPS_PER_ROLLOUT //  MINIBATCH_SIZE

###############################################################################

class PPOAgent:
    def __init__(self, env):
        # Environment
        self.env = env
        self.env_action_ub = env.action_space.high[0]
        self.env_action_lb = env.action_space.low[0]
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.shape[0]
        # Actor
        self.actor = Actor(self.state_size, self.action_size)
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=ACTOR_LEARNING_RATE)
        # Critic
        self.critic = self.create_critic()
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=CRITIC_LEARNING_RATE)

    def create_critic(self):
        inputs_state = layers.Input(shape=(self.state_size,))
        hidden1 = layers.Dense(256, activation="relu")(inputs_state)
        hidden2 = layers.Dense(256, activation="relu")(hidden1)
        output = layers.Dense(1)(hidden2)

        model = tf.keras.Model(inputs=inputs_state, outputs=output)
        return model


def test_env(agent):
    total_rewards = []
    for i in range(10):
        state = tf.expand_dims(tf.convert_to_tensor(agent.env.reset()), 0)
        done = False
        ep_rewards = 0
        while not done:
            state = tf.reshape(state, [1, agent.state_size])
            dist = agent.actor(state)
            action = dist.mean()
            state, reward, done, _ = env.step(action)
            ep_rewards += reward[0]
        total_rewards.append(ep_rewards)
    return np.mean(total_rewards)


def calculate_advantages(critic_model, reward_history, value_history, final_state, reached_done):
    # Put inputs into correct format
    reward_history = tf.reshape(reward_history, [np.shape(reward_history)[0]])
    # Get bootstrapped value for the n+1 step\
    # print("===============================")
    # print("INPUT STATE!!!: ", final_state)
    # print("===============================")
    bootstrapped_value = critic_model(final_state)
    bootstrapped_value = tf.reshape(bootstrapped_value, [1])
    final_value = tf.expand_dims(0.0, 0) if reached_done else bootstrapped_value


    # Calculate critic values V(s) for each state in the trajectory
    # print("VALUE HISTORY", value_history)
    value_history = tf.reshape(value_history, [np.shape(value_history)[0]])
    # print("VALUE HISTORY", value_history)
    # print("final valueee", final_value)
    value_history = tf.concat([value_history, final_value], 0)

    # Calculate advantages
    # deltas = reward_history + GAMMA * value_history[1:] - value_history[:-1]
    # advantages = []
    # gae = 0
    # for d in deltas[::-1]:
    #     gae = d + GAMMA * LAMBDA * gae
    #     advantages.append(tf.expand_dims(gae, 0))
    # advantages.reverse()
    #
    # returns = []
    # ret = 0
    # for r in reward_history[::-1]:
    #     ret = r + GAMMA * ret
    #     returns.append(tf.expand_dims(ret, 0))
    # returns.reverse()

    # This is another way to do it, but now returns also uses GAE
    gae = 0
    returns = []
    for step in reversed(range(len(reward_history))):
        delta = reward_history[step] + GAMMA * value_history[step + 1]  - value_history[step]
        gae = delta + GAMMA * LAMBDA  * gae
        returns.insert(0, gae + value_history[step])
    advantages = returns - value_history[:-1]

    return advantages, returns

def run_rollout(agent, replay_buffer, num_rollout_steps=NUM_STEPS_PER_ROLLOUT):
    """
    Run policy pi_old in environment for T timesteps.
    Compute advantage estimates A_1, ..., A_T
    """
    cur_step = 0
    while (cur_step < num_rollout_steps):
        state_trajectory = []
        action_trajectory= []
        reward_trajectory = []
        mask_trajectory = []
        value_trajectory = []
        logp_trajectory = []

        state = tf.expand_dims(agent.env.reset(), 0)
        # print("FIRST EVER STATE: ", state)
        done = False
        while (cur_step < num_rollout_steps and not done):
            # Pass state through actor and critic
            # print("===============================")
            # print("INPUT STATE!!!: ", state)
            # print("===============================")

            pi_dist = agent.actor(state)
            action = pi_dist.sample()
            logp = pi_dist.log_prob(action)

            critic_value = agent.critic(state)

            # Step
            next_state, reward, done, _ = agent.env.step(action)
            next_state = tf.reshape(next_state, [1, agent.state_size])

            # Bookkeeping
            state_trajectory.append(state)
            action_trajectory.append(action)
            reward_trajectory.append(tf.convert_to_tensor(reward))
            mask_trajectory.append(tf.convert_to_tensor([1 - done]))

            value_trajectory.append(tf.reshape(critic_value, [np.shape(critic_value)[0]]))
            logp_trajectory.append(tf.reshape(logp, [np.shape(logp)[0]]))

            state = next_state
            cur_step += 1

        # Calculate A & G
        advantages, returns = calculate_advantages(agent.critic,
                                                   reward_trajectory,
                                                   value_trajectory,
                                                   state,
                                                   done)

        # print(f"===\nSTATE_TRAJECTORY:: {state_trajectory}\n===")
        # print(f"===\nACTION_TRAJECTORY:: {action_trajectory}\n===")
        # print(f"===\nREWARD_TRAJECTORY:: {reward_trajectory}\n===")
        # print(f"===\nMASK_TRAJECTORY:: {mask_trajectory}\n===")
        # print(f"===\nVALUE_TRAJECTORY:: {value_trajectory}\n===")
        # print(f"===\nLOGP_TRAJECTORY:: {logp_trajectory}\n===")
        # print(f"===\nADVANTAGES:: {advantages}\n===")
        # print(f"===\nRETURNS:: {returns}\n===")

        # Store in buffer
        batch_transitions = zip(state_trajectory, action_trajectory,
                                reward_trajectory, mask_trajectory,
                                value_trajectory, logp_trajectory,
                                advantages, returns)
        for s, a, r, m, v, logp, adv, ret in batch_transitions:
            # print("S:", s)
            # print("A:", a)
            # print("R:", r)
            # print("M:", m)
            # print("V:", v)
            # print("LOGP:", logp)
            # print("ADV:", adv)
            # print("RET:", ret)
            replay_buffer.store_transition((s, a, r, m, v, logp, adv, ret))


def run_epoch(agent, replay_buffer, num_steps_per_rollout):
    # Step 1: Rollout policy T steps
    # Step 2: Calculate advantages and G
    # Step 3: Store in buffer
    ep_rewards = run_rollout(agent, replay_buffer, num_steps_per_rollout)

    # Do multiple gradient updates with the same rollout data!
    for i in range(k_train_iters):
        # Step 4: Retrieve random minibatch of transitions of size M
        batch_transitions = replay_buffer.sample()
        states, actions, rewards, masks, values, \
            logps_old, advs, rets = batch_transitions
        # print("S:", states)
        # print("A:", actions)
        # print("R:", rewards)
        # print("M:", masks)
        # print("V:", values)
        # print("LOGP:", logps_old)
        # print("ADV:", advs)
        # print("RET:", rets)
        logps_old = tf.cast(tf.reshape(logps_old, [np.shape(logps_old)[0]]), tf.float32)
        advs = tf.cast(tf.reshape(advs, [np.shape(advs)[0]]), tf.float32)
        # print("------------------------------------------------------------")
        # print("LOGP:", logps_old)
        # print("ADV:", advs)
        states = tf.cast(states, tf.float32)
        actions = tf.cast(actions, tf.float32)

        # Step 5: Calculate Actor Loss and do gradient step
        with tf.GradientTape() as tape:
            # Step 5.1, get pi_distribution
            pi_dists = agent.actor(states)
            # Step 5.2, get log_prob of action taken
            logps_new = pi_dists.log_prob(actions)
            ratios = tf.math.exp(logps_new - logps_old)
            surr1 = ratios * advs
            surr2 = tf.clip_by_value(ratios, 1.0 - CLIP_PARAM, 1.0 + CLIP_PARAM) * advs
            actor_loss = -tf.reduce_mean(tf.math.minimum(surr1, surr2))
            entropy_loss = -tf.reduce_mean(pi_dists.entropy()) * ENTROPY_WEIGHT
            actor_loss += entropy_loss

        # Step 5.3: Use KL to determine whether we should break out or not
        kl = tf.reduce_mean(logps_old - logps_new)
        # if kl > 1.5 * TARGET_KL:
        #     print(f"Breaking out at iter {round(i, 3)}/{k_train_iters} because KL {kl} > {1.5 * TARGET_KL}")
        #     break
        grads = tape.gradient(actor_loss, agent.actor.trainable_variables)
        agent.actor_optimizer.apply_gradients(zip(grads, agent.actor.trainable_variables))

    for _ in range(k_train_iters):
        # Step 6: Retrieve random minibatch of transitions of size M
        batch_transitions = replay_buffer.sample()
        states, actions, rewards, dones, values, \
            logps_old, advs, rets = batch_transitions

        # Step 7: Calculate Critic Loss and do gradient step
        with tf.GradientTape() as tape:
            values = agent.critic(states)
            critic_loss = tf.reduce_mean(tf.square(values - rets))
        grads = tape.gradient(critic_loss, agent.critic.trainable_variables)
        agent.critic_optimizer.apply_gradients(zip(grads, agent.critic.trainable_variables))

    replay_buffer.flush()

    return ep_rewards


if __name__ == '__main__':
    env = gym.make("Pendulum-v0")
    agent = PPOAgent(env)
    replay_buffer = PPOBuffer(state_size=agent.state_size,
                              action_size=agent.action_size,
                              capacity=NUM_STEPS_PER_ROLLOUT,
                              batch_size=MINIBATCH_SIZE)

    # Run training
    running_reward = 0
    for e in range(NUM_EPISODES):
        run_epoch(agent, replay_buffer, NUM_STEPS_PER_ROLLOUT)

        if e % 10 == 0:
            avg_ep_rews = test_env(agent)
            running_reward = 0.05 * avg_ep_rews + (1 - 0.05) * running_reward
            template = "running reward: {:.2f} | episode reward: {:.2f} at episode {}"
            print(template.format(running_reward, avg_ep_rews, e))

        # if running_reward > -140:
        #     print("Solved at episode {}!".format(e))
        #     break
