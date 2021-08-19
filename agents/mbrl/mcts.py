"""
Monte Carlo Tree Search
Alternate the following 4 steps.
1) Select: Using a TreePolicy, select an action given the current state. We do this until we get to a leaf.
2) Expand: When we get to a leaf, we should expand if it A) isn't a terminal node and B) hasn't been fully
            expanded yet.
3) Simulate: For the leaf node, use the DefaultPolicy (e.g. a random policy) to do a rollout until the terminal state.
                Sum the rewards of the rollout.
4) Back-up: After completing a rollout, update the total observed rewards (T) and count (N) of the current Node.
                Then back these values up to the parent until reaching the root Node
"""
import argparse
from copy import deepcopy
import numpy as np
from typing import List, Union, Tuple
import gym
import time


class MCTSNode:
    def __init__(self,
                 parent: 'MCTSNode' = None,
                 parent_action: np.ndarray = None,
                 env_instance: gym.Env = None,
                 state: np.ndarray = None,
                 immediate_reward: Union[float, int] = 0.,
                 is_terminal: bool = False):
        self.parent = parent
        self.parent_action = parent_action
        self.N = 0  # The number of times this Node has been visited
        self.T = 0  # The total sum of rewards from visiting this Node
        self.children = []
        self.is_expanded = False

        if self.parent is None:
            self.state = env.reset() if state is None else state
            self.env = deepcopy(env) if state is None else env_instance
            self.immediate_reward = 0
            self.is_terminal = False
        else:
            self.state = state
            self.env = deepcopy(env_instance)
            self.immediate_reward = immediate_reward
            self.is_terminal = is_terminal

        self._env_copy = deepcopy(self.env)

    @property
    def is_leaf(self) -> bool:
        """The Node is a leaf is it has no children."""
        return len(self.children) == 0

    @property
    def is_root(self) -> bool:
        """The Node is the root if it has no parent."""
        return self.parent is None

    @property
    def V(self) -> float:
        """Compute the average sum of rewards from this Node onward (i.e. V(s))"""
        return self.T / self.N if self.N > 0 else 0

    @property
    def ucb(self) -> float:
        """Compute the Upper Confidence Bound"""
        if self.N == 0:
            return float('inf')
        return self.V + 2 * np.sqrt(2 * np.log(2 * self.parent.N / self.N))

    def reset_env(self) -> None:
        self.env = deepcopy(self._env_copy)

    def select(self) -> 'MCTSNode':
        """
        Upper Confidence Tree (UCT) search.
        Recursively traverse down the tree to find a leaf node by selecting the
        child Node with the highest UCB score.
        If the current Node is a leaf itself, return itself.

        :return: Selected leaf Node
        """
        if self.is_leaf:
            return self
        assert self.is_expanded is True, "You must be expanded before you can select a leaf node"

        ucbs = [child.ucb for child in self.children]
        best_child_idx = np.argmax(ucbs)
        best_child = self.children[best_child_idx]
        return best_child.select()

    def expand(self) -> 'MCTSNode':
        """
        Expand the tree by creating children Nodes for all possible actions from the current state.
        Then randomly select and return one of those children.
        :return:
        """
        assert self.is_expanded is False, "You've already expanded this Node. You cannot expand again!"
        assert self.is_terminal is False, "This Node is terminal. You cannot expand a terminal Node!"

        self.is_expanded = True
        possible_actions = get_available_actions(environment=self.env, state=self.state)
        for a in possible_actions:
            self.reset_env()  # We want to reset the env everytime to the Node's state
            next_state, reward, done, _ = self.env.step(a)

            child = MCTSNode(parent=self, parent_action=a, env_instance=self.env, state=next_state,
                             immediate_reward=reward, is_terminal=done)
            self.children.append(child)
        return self.select()

    def rollout(self) -> float:
        """
        Complete 1 rollout starting from the current state. Take random actions
        until reaching a terminal node.
        :return: total sum of rewards from rollout
        """
        self.reset_env()  # We want to reset the env everytime to the Node's state
        rollout_reward = 0
        done = self.is_terminal
        available_actions = get_available_actions(environment=self.env, state=self.state)
        while not done:
            action = np.random.choice(available_actions)
            state, reward, done, _ = self.env.step(action)
            rollout_reward += reward
        return rollout_reward

    def backpropagate(self, reward: float) -> None:
        """
        Propagate this Node's total sum of rewards T back up to its parent.
        Do this recursively until reaching the root Node.

        :param reward:
        :return:
        """
        self.N += 1
        self.T += reward + self.immediate_reward  # Should I also do + self.immediate_reward?
        if not self.is_root:
            self.parent.backpropagate(reward=reward)


def _tree_policy(node: 'MCTSNode') -> 'MCTSNode':
    """
    Policy for finding and return a leaf Node.
    :param node: Node to begin the policy
    :return: a leaf Node
    """
    while not node.is_terminal:
        if not node.is_expanded:
            return node.expand()
        else:
            node = node.select()
            # If the Node hasn't been attempted yet, return
            # This isn't the most elegant way to enforce this, but it's easiest to
            # understand in my opinion.
            if node.N == 0:
                return node
    return node


def _default_policy(node: 'MCTSNode') -> float:
    """
    Complete 1 rollout starting at `node`.
    :param node: starting Node
    :return: total sum of rewards from rollout
    """
    return node.rollout()


def _select_best_child(node: 'MCTSNode') -> 'MCTSNode':
    """
    Given the current Node, select the child with the maximum expected rewards.
    :param node: current Node
    :return: the action with the maximum expected rewards
    """
    values = [child.V for child in node.children]
    best_child_idx = np.argmax(values)
    return node.children[best_child_idx]


def select_best_action(root: 'MCTSNode', simulation_steps: int) -> np.ndarray:
    """
    Run the MCTS algorithm for one iteration. This will make 1 update of the tree.
    :param root: The root Node of the tree.
    :param simulation_steps: The number of steps to run the MCTS algorithm before selecting an action
    :return: the action with the best expected rewards
    """
    for _ in range(simulation_steps):
        leaf = _tree_policy(root)
        rollout_rewards = _default_policy(leaf)
        leaf.backpropagate(reward=rollout_rewards)
    best_child = _select_best_child(root)
    return best_child.parent_action


def run_episode(env: gym.Env, simulation_steps: int) -> Tuple[List, int]:
    initial_state = env.reset()
    root = MCTSNode(parent=None, parent_action=None, env_instance=env, state=initial_state)

    ep_rewards = []
    cur_step = 0
    done = False
    while not done:
        # Select the best action and take a step
        action = select_best_action(root, simulation_steps)
        next_state, reward, done, _ = env.step(action)

        # Restart at the new state
        root = MCTSNode(parent=root, parent_action=action, env_instance=env, state=next_state,
                        immediate_reward=reward, is_terminal=done)

        # Some bookkeeping
        ep_rewards.append(reward)
        cur_step += 1
    return ep_rewards, cur_step


def get_available_actions(environment: gym.Env, state: np.ndarray) -> List:
    """
    Based on the current state, return a List of all valid actions.
    In some environments this may depend on the state which is why
    I've included it as a param even though it's not currently being used.
    Currently, this just supports discrete action settings.
    """
    assert isinstance(environment.action_space, gym.spaces.Discrete), (
        "This currently supports discrete action spaces only!")
    return [_ for _ in range(environment.action_space.n)]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="CartPole-v1")
    parser.add_argument("--mcts_simulation_steps", type=int, default=100)  # 20 for CartPole-v0, 100 for CartPole-v1
    parser.add_argument("--seed", type=int, default=1)

    args = parser.parse_args()

    # Create environment
    env = gym.make(args.env)

    # Set seeds
    if args.seed:
        np.random.seed(args.seed)
        env.seed(args.seed)

    # Run MCTS planning and run one trajectory
    start = time.time()
    ep_rew, ep_steps = run_episode(env=env, simulation_steps=args.mcts_simulation_steps)
    elapsed_time = round(time.time() - start, 2)
    print("MCTS Simulation steps: ", args.mcts_simulation_steps)
    print("Elapsed time (sec): ", elapsed_time)
    print("Episode rewards: ", sum(ep_rew))
    print("Episode num steps: ", ep_steps)
