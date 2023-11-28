import random
from typing import Callable, Tuple

import gym
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch import distributions

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(f"Use cuda: {use_cuda} -- device: {device}")

# Environment constants
SEED = 0
ENV_NAME = "MountainCarContinuous-v0"
NUM_ENVS = 16  # number of parallel environments N

# Network constants
HIDDEN_SIZE = 256
LEARNING_RATE = 5e-4

# Learning constants
CLIP_PARAM = 0.2  # epsilon
ENTROPY_WEIGHT = 0.001

# Batch info
NUM_STEPS_PER_ENV = 1024  # num of transitions T we sample for each training iter
BATCH_SIZE = NUM_ENVS * NUM_STEPS_PER_ENV
MINIBATCH_SIZE = 16  # num of samples randomly selected from stored data
NUM_ITER_PER_BATCH = 8  # number of passes through the batch

# Other constants
EPOCHS = 20  # num of training rounds
THRESHOLD = 90


def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        # delta is Bellman equation minus value of the state
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        # moving average of advantages discounted by gamma * tau
        gae = delta + gamma * tau * masks[step] * gae
        returns.insert(0, gae + values[step])
    return returns


def make_env(env_name: str, seed: int) -> Callable[[], gym.Env]:
    """Given an environment name, return a function that can be called to create an environment."""

    def _thunk() -> gym.Env:
        env = gym.make(env_name)
        env.seed(seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
        return env

    return _thunk


class ActorCritic(nn.Module):
    pass


def evaluate_policy(render=False):
    state = eval_env.reset()
    if render:
        eval_env.render()
    done = False
    total_reward = 0
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist, _ = model(state)
        next_state, reward, done, _ = eval_env.step(dist.sample().cpu().numpy()[0])
        state = next_state
        if render:
            eval_env.render()
        total_reward += reward
    return total_reward


def sample_transitions(mini_batch_size, states, actions, log_probs, returns, advantage) -> Tuple:
    # gather random indicies

    # return transitions from those indices
    pass


def rollout_policy(num_steps_per_env: int):
    # initialize all the data structures you need

    # Rollout policy for T timesteps
    for _ in range(num_steps_per_env):
        # Store log_probs, values, rewards, done_masks, states, actions.

        # Each list num_steps long, each step num_envs wide.

        pass

    # After completing the rollout, calculate the returns
    # to calc returns correctly, run final next_state through network to get value

    # Run GAE. Loop backwards from recent experience.

    # concatanate each list inside a torch tensor.
    # list that was num_steps long, num_envs wide becomes num_steps*num_envs long

    pass


def train_episode():
    # Rollout the policy

    # Iterate over the batch multiple times
    for _ in range(NUM_ITER_PER_BATCH):
        # Complete 1 pass over the Batch in increments of Minibatch size.
        for _ in range(BATCH_SIZE // MINIBATCH_SIZE):
            # sample transitions

            # Calculate loss

            # Update network

            pass


if __name__ == "__main__":
    # Set random seeds
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.backends.cudnn.deterministic = True

    # Create environments
    envs = gym.vector.SyncVectorEnv([make_env(ENV_NAME, SEED + i) for i in range(NUM_ENVS)])
    eval_env = gym.make(ENV_NAME)  # for eval only
    eval_env.seed(SEED)

    num_inputs = envs.single_observation_space.shape[0]
    num_outputs = envs.single_action_space.shape[0]

    # Create model
    model = ActorCritic(num_inputs, num_outputs, HIDDEN_SIZE).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Run training
    eval_rewards = []
    for e in range(EPOCHS):
        train_episode()

        # Evaluate agent
        eval_rews = np.mean([evaluate_policy() for _ in range(10)])
        eval_rewards.append(eval_rews)
        print(f"Epoch: {e} | Reward: {round(eval_rews, 2)}")
        if eval_rews >= THRESHOLD:
            break

    plt.title(f"PPO - {ENV_NAME}")
    plt.plot([i for i in range(len(eval_rewards))], eval_rewards)
    plt.xlabel("Epoch")
    plt.ylabel("Avg rewards")
    plt.show()

    # evaluate_policy(render=True)
