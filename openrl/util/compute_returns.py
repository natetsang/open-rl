from typing import List, Union
import numpy as np


def compute_gae_returns(next_value: np.ndarray,
                        rewards: List,
                        masks: List,
                        values: List,
                        gamma: float = 0.99,
                        lambda_: float = 0.95) -> List:
    """
    Computes the generalized advantage estimation (GAE) of the returns.
    @source https://towardsdatascience.com/proximal-policy-optimization-tutorial-part-2-2-gae-and-ppo-loss-fe1b3c5549e8

    :param next_value:
    :param rewards:
    :param masks:
    :param values:
    :param gamma:
    :param lambda_:
    :return: GAE of the returns
    """
    values = values + [next_value]
    gae = 0
    returns = []
    for step in reversed(range(len(rewards))):
        delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
        gae = delta + gamma * lambda_ * masks[step] * gae

        # Notice I'm adding back the value to get the reward
        returns.insert(0, gae + values[step])

    return returns


def compute_bootstrapped_returns(rewards: List, values: List, gamma: float = 0.95) -> List:
    """
    Compute bootstrapped rewards-to-go. It's assumed the last state is the terminal state,
    so V(s_T) = 0.

    q(s_t, a_t) = r(s_t, a_t) + V(s_t+1) * (1 - done)

     :param rewards:
     :param values:
     :param gamma:
     :return:
     """
    returns = []
    for step in range(len(rewards) - 1):
        q_t = rewards[step] + gamma * values[step + 1]
        returns.append(q_t)
    returns.append(rewards[-1])  # terminal state -> V(s_T) = 0
    return returns


def compute_discounted_returns(next_value: Union[float, np.ndarray], rewards: List, masks: List,
                               gamma: float = 0.95) -> List:
    """
    :param next_value:
    :param rewards:
    :param masks:
    :param gamma:
    :return:
    """
    discounted_rewards = []
    total_ret = next_value * masks[-1]
    for r in rewards[::-1]:
        total_ret = r + gamma * total_ret
        discounted_rewards.insert(0, total_ret)
    return discounted_rewards


def compute_undiscounted_returns(next_value: Union[float, np.ndarray], rewards: List, masks: List) -> List:
    return compute_discounted_returns(next_value, rewards, masks, gamma=1)


def compute_returns_simple(rewards: List, gamma: float = 0.95) -> List:
    """
    Compute the rewards-to-go, which are the cumulative rewards from t=t' to T.

    :param rewards: a list of rewards where the ith entry is the reward received at timestep t=i.
    :param gamma: discount factor between 0 and 1.0 (where 1.0 means it's not discounted)
    :return: the rewards-to-go, where the ith entry is the cumulative rewards from timestep t=i to t=T,
        where T is equal to len(rewards).
    """
    discounted_rewards = []
    total_ret = 0
    for r in rewards[::-1]:
        # Without discount
        # total_ret = r + total_ret

        # With discount
        total_ret = r + gamma * total_ret
        discounted_rewards.insert(0, total_ret)
    return discounted_rewards
