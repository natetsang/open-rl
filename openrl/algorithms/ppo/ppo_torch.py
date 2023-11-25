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
SEED = 1
ENV_NAME = "MountainCarContinuous-v0"
NUM_ENVS = 16

# Network constants
HIDDEN_SIZE = 256
LEARNING_RATE = 3e-4

# Learning constants
CLIP_PARAM = 0.2
ENTROPY_WEIGHT = 0.001
TARGET_KL = 0.015

# Other constants
NUM_STEPS_PER_ENV = 1024  # num of transitions we sample for each training iter
BATCH_SIZE = NUM_ENVS * NUM_STEPS_PER_ENV
MINIBATCH_SIZE = 16  # num of samples randomly selected from stored data
EPOCHS = 16  # num passes over entire training data
NUM_ITER_PER_BATCH = 8
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
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
        super(ActorCritic, self).__init__()

        self.actor = nn.Sequential(
            self.layer_init(nn.Linear(num_inputs, hidden_size)),
            nn.Tanh(),
            self.layer_init(nn.Linear(hidden_size, num_outputs)),
        )

        self.critic = nn.Sequential(
            self.layer_init(nn.Linear(num_inputs, hidden_size)),
            nn.Tanh(),
            self.layer_init(nn.Linear(hidden_size, 1), std=1.0),
        )

        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)

    def layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        nn.init.orthogonal_(layer.weight, std)
        nn.init.constant_(layer.bias, bias_const)
        return layer

    def forward(self, x):
        value = self.critic(x)
        mu = self.actor(x)
        std = self.log_std.exp().expand_as(mu)  # make log_std the same shape as mu
        dist = distributions.Normal(mu, std)
        return dist, value


def evaluate_policy(render=False):
    state = env.reset()
    if render:
        env.render()
    done = False
    total_reward = 0
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        dist, _ = model(state)
        next_state, reward, done, _ = env.step(dist.sample().cpu().numpy()[0])
        state = next_state
        if render:
            env.render()
        total_reward += reward
    return total_reward


def sample_transitions(mini_batch_size, states, actions, log_probs, returns, advantage) -> Tuple:
    rand_ids = np.random.randint(0, len(states), mini_batch_size)

    return (
        states[rand_ids, :],
        actions[rand_ids, :],
        log_probs[rand_ids, :],
        returns[rand_ids, :],
        advantage[rand_ids, :],
    )


def rollout_policy(num_steps_per_env: int):
    states = []
    actions = []
    rewards = []
    masks = []
    values = []
    log_probs = []

    state = envs.reset()

    for _ in range(num_steps_per_env):
        state = torch.FloatTensor(state).to(device)
        dist, value = model(state)  # state through netwwork to get prob dist and estimated V(s)

        action = dist.sample()
        next_state, reward, done, _ = envs.step(action.cpu().numpy())
        log_prob = dist.log_prob(action)

        # Store log_probs, values, rewards, done_masks, states, actions.
        # Each list num_steps long, each step num_envs wide.
        states.append(state)
        actions.append(action)
        rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
        masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))
        values.append(value)
        log_probs.append(log_prob)

        state = next_state

    # After completing the rollout, calculate the returns
    # to calc returns correctly, run final next_state through network to get value
    next_state = torch.FloatTensor(next_state).to(device)
    _, next_value = model(next_state)
    # run GAE. Loop backwards from recent experience.
    returns = compute_gae(next_value, rewards, masks, values)

    # concatanate each list inside a torch tensor.
    # list that was num_steps long, num_envs wide becomes num_steps*num_envs long
    returns = torch.cat(returns).detach()
    log_probs = torch.cat(log_probs).detach()
    values = torch.cat(values).detach()
    states = torch.cat(states)
    actions = torch.cat(actions)
    advantages = returns - values
    return states, actions, rewards, masks, values, log_probs, returns, advantages


def train_episode():
    # Step 1: Rollout policy and calculate Advantages and Returns
    states, actions, _, _, _, log_probs, returns, advantages = rollout_policy(NUM_STEPS_PER_ENV)

    batch_size = states.size(0)
    for _ in range(NUM_ITER_PER_BATCH):
        # Complete 1 pass-through of the Batch in increments of Minibatch size.
        for _ in range(batch_size // MINIBATCH_SIZE):
            state, action, old_log_probs, return_, advantage = sample_transitions(
                MINIBATCH_SIZE, states, actions, log_probs, returns, advantages
            )

            dist, value = model(state)

            # Calculate loss
            entropy = dist.entropy().mean()  # for inciting exploration
            new_log_probs = dist.log_prob(action)  # new log_probs of originally selected actions
            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - CLIP_PARAM, 1.0 + CLIP_PARAM) * advantage

            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = (return_ - value).pow(2).mean()
            loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

            # with torch.no_grad():
            #     kl = ((ratio - 1) - (new_log_probs - old_log_probs)).mean()
            #     if kl > TARGET_KL:
            #         break

            # Update network
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


if __name__ == "__main__":
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(42)
    torch.backends.cudnn.deterministic = True

    envs = gym.vector.SyncVectorEnv([make_env(ENV_NAME, SEED + i) for i in range(NUM_ENVS)])
    env = gym.make(ENV_NAME)  # for eval only

    num_inputs = envs.single_observation_space.shape[0]
    num_outputs = envs.single_action_space.shape[0]

    model = ActorCritic(num_inputs, num_outputs, HIDDEN_SIZE).to(device)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Run training
    eval_rewards = []
    for e in range(EPOCHS):
        train_episode()

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
