from modified_tensorboard import ModifiedTensorBoard
from blob import Blob
from blob_env import BlobEnv

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.optimizers import Adam
from collections import deque
import time
import random
import os
import numpy as np
from tqdm import tqdm

# For more repetitive results
random.seed(1)
np.random.seed(1)
tf.random.set_seed(1)

DISCOUNT = 0.99
REPLAY_MEMORY_SIZE = 50_000  # How many last steps to keep for model training
MIN_REPLAY_MEMORY_SIZE = 1_000  # Minimum number of steps in a memory to start training
MINIBATCH_SIZE = 64  # How many steps (samples) to use for training
UPDATE_TARGET_EVERY = 5  # Terminal states (end of episodes)
MODEL_NAME = '2x256'
MIN_REWARD = -200  # For model save
MEMORY_FRACTION = 0.20

# Environment settings
EPISODES = 20

# Exploration settings
epsilon = 1  # not a constant, going to be decayed
EPSILON_DECAY = 0.99975
MIN_EPSILON = 0.001

#  Stats settings
AGGREGATE_STATS_EVERY = 50  # episodes
SHOW_PREVIEW = True  # Set to true if we want to see visuals of everything running


class DQNAgent:
    """
    Why do we have two models?
    All NN are randomly initialized. Also, our agent will likely start with
    an episilon of 1, meaning that will take purely random actions.

    Initially, this model will be trying to fit to a bunch of random things,
    which is useless. As we explore the environment, the model will become more useful.

    Since we're doing a .predict() every single step the agent takes, and we want
    to have some consistency in the .predicts(). We're also doing a .fit() every
    single step. So this model will initially be all over the place as it attempts
    to figure things out randomly. We're continually trying to fit to a ever-
    changing model. The model is kind of chasing its own tail. This is bad.

    We will avoid this by having two models.
    self.model ==> we .fit() every step (gets trained every step)
    self.target_model ==> we .predict() every step

    By predicting on the target model, the agent is taking steps that are consistent
    with a model that's not changing as much over time. Kind of hard to explain...

    Then every n steps/episodes, we will copy the weights from the main to the target model.
    By having a target model, we give our predictions some consistency/stability
    so our model can actually learn something.

    Replay memory:
    This will help us reduce the issues with calling .fit() every single step.
    Typically in NN we call .fit() on a batch of records. Here, we call .fit()
    on only one record. We typically train on a batch because it's quicker and
    it tends to have better results and a more stable model. Thus, it won't overfit
    to one sample. If we think about gradient descent, we typically use batches or mini-batches.
    We we take only one sample, the gradient zig-zags a lot because it's susceptible
    to the variation in that one sample.

    So we will take a random batch within the replay memory.
    """

    def __init__(self):
        # main model
        self.model = self.create_model()

        # target model
        self.target_model = self.create_model()
        self.target_model.set_weights(self.model.get_weights())

        self.replay_memory = deque(maxlen=REPLAY_MEMORY_SIZE)
        self.tensorboard = ModifiedTensorBoard(log_dir=f"logs/{MODEL_NAME}-{int(time.time())}")

        # We use this to track internally when we're going to update our target model
        self.target_update_counter = 0

    def create_model(self):
        model = Sequential()

        model.add(Conv2D(filters=256, kernel_size=3, input_shape=env.OBSERVATION_SPACE_VALUES, activation="relu"))
        model.add(MaxPooling2D(2))
        model.add(Dropout(0.2))

        model.add(Conv2D(filters=256, kernel_size=3, activation="relu"))
        model.add(MaxPooling2D(2))
        model.add(Dropout(0.2))

        model.add(Flatten())
        model.add(Dense(64))
        model.add(Dense(env.ACTION_SPACE_SIZE, activation="linear"))

        model.compile(loss="mse", optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
        return model

    def update_replay_memory(self, transition):
        # Transition = (s, a, r, s')
        self.replay_memory.append(transition)

    def get_qs(self, state):
        # model.predict() always returns a list
        # we divide by 255 b/c we want to normalize the RGB data
        # *state.shape just unpacks state
        return self.model.predict(np.array(state).reshape(-1, *state.shape) / 255)[0]


    def train(self, terminal_state, step):
        """We want our batch size to be pretty small compared to our memory.
        We don't want to have replay memory so small so we're training on the
        same data.

        We train with a minibatch every time train() is called."""

        # We don't have enough sample in our replay memory, don't train yet.
        if len(self.replay_memory) < MIN_REPLAY_MEMORY_SIZE:
            return

        # minibatch is a list of transitions
        # each transition is a tuple (current_state, action, reward, new_current_state, done)
        minibatch = random.sample(self.replay_memory, MINIBATCH_SIZE)

        # Get current states and predict qs based on main model
        # Since transition = (s, a, r, s'), transition[0] is the current state
        current_states = np.array([transition[0] for transition in minibatch]) / 255
        current_qs_list = self.model.predict(current_states)  # less stable model

        # This is after we took the step
        # Get next states and predict qs based on the target model
        # Since transition = (s, a, r, s'), transition[3] is the next state
        new_current_states = np.array([transition[3] for transition in minibatch]) / 255
        future_qs_list = self.target_model.predict(new_current_states)  # more stable model

        X = list()  # images from the game
        y = list()  # predicted q-values given the state and the action

        """
        Now we go through each transition in the minibatch, and fit.
        """
        for index, (current_state, action, reward, new_current_state, done) in enumerate(minibatch):
            if not done:
                max_future_q = np.max(future_qs_list[index])
                new_q = reward + DISCOUNT * max_future_q
            else:
                # If done, there is no future q, because we are done
                new_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = new_q

            X.append(current_state)  # the image that we have
            y.append(current_qs)  # the current qs for this image

        # I don't think we actually need to divide np.array(X) by 255 because we already
        # do that above.
        self.model.fit(np.array(X) / 255, np.array(y), batch_size=MINIBATCH_SIZE,
            verbose=0, shuffle=False, callbacks=[self.tensorboard] if terminal_state else None)

        # updating to determine if we want to update target_model yet
        if terminal_state:
            self.target_update_counter += 1

        if self.target_update_counter > UPDATE_TARGET_EVERY:
            # Copy weights from main weights to target weights
            self.target_model.set_weights(self.model.get_weights())
            self.target_update_counter = 0

env = BlobEnv()
agent = DQNAgent()

# For stats
ep_rewards = [-200]

# Create models folder
if not os.path.isdir('models'):
    os.makedirs('models')

# tqmd is just a package that shows a progress bar in the CLI as it runs
for episode in tqdm(range(1, EPISODES + 1), ascii=True, unit="episode"):
    agent.tensorboard.step = episode

    episode_reward = 0
    step = 1
    current_state = env.reset()

    done = False

    while not done:
        # Take action given current state
        if np.random.random() > epsilon:
            action = np.argmax(agent.get_qs(current_state))
        else:
            action = np.random.randint(0, env.ACTION_SPACE_SIZE)

        new_state, reward, done = env.step(action)

        episode_reward += reward

        if SHOW_PREVIEW and not episode % AGGREGATE_STATS_EVERY:
            env.render()

        # add transition to replay memory
        agent.update_replay_memory((current_state, action, reward, new_state, done))
        agent.train(done, step)

        current_state = new_state
        step += 1

    # Append episode reward to a list and log stats (every given number of episodes)
    ep_rewards.append(episode_reward)
    # Aggregate stats and put them into tensorboard. We could have put it in matplotlib if you wanted
    if not episode % AGGREGATE_STATS_EVERY or episode == 1:
        average_reward = sum(ep_rewards[-AGGREGATE_STATS_EVERY:])/len(ep_rewards[-AGGREGATE_STATS_EVERY:])
        min_reward = min(ep_rewards[-AGGREGATE_STATS_EVERY:])
        max_reward = max(ep_rewards[-AGGREGATE_STATS_EVERY:])
        agent.tensorboard.update_stats(reward_avg=average_reward, reward_min=min_reward, reward_max=max_reward, epsilon=epsilon)

        # Find a good reason to save a model. Don't just save the same model over and over
        # Save model, but only when min reward is greater or equal a set value
        if min_reward >= MIN_REWARD:
            agent.model.save(f'models/{MODEL_NAME}__{max_reward:_>7.2f}max_{average_reward:_>7.2f}avg_{min_reward:_>7.2f}min__{int(time.time())}.model')

    # Decay epsilon
    if epsilon > MIN_EPSILON:
        epsilon *= EPSILON_DECAY
        epsilon = max(MIN_EPSILON, epsilon)
