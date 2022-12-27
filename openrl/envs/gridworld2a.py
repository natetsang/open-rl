import gym
import numpy as np
from gym import spaces, utils


class GridWorld(gym.Env):
    def __init__(self, noise: float = 0.2, living_reward: float = -0.1):
        self.num_rows = 3
        self.num_cols = 4
        self.action_to_direction_map = {
            0: (-1, 0),  # N
            1: (1, 0),  # S
            2: (0, -1),  # W
            3: (0, 1),  # E
        }
        self.action_space = spaces.Discrete(len(self.action_to_direction_map))

        self.observation_space = spaces.Dict(
            {
                "agent": spaces.MultiDiscrete([self.num_rows, self.num_cols]),
                "goal": spaces.MultiDiscrete([self.num_rows, self.num_cols]),
            }
        )
        self.noise = noise

        # Rewards
        self.living_reward = living_reward
        self.goal_reward = 1
        self.terminal_reward = -1

        self.start_pos = (2, 0)
        self.goal_pos = (0, 3)
        self.terminal_pos = (1, 3)
        self.obstacle_pos = (1, 1)

        self.agent_pos = self.start_pos

        # Rendering
        self.rendering_grid = np.asarray(
            [
                "OOOG",
                "OWOT",
                "OOOO",
            ],
            dtype="c",
        )

    def _get_observation(self) -> dict[str, tuple[int, int]]:
        return {"agent": self.agent_pos, "goal": self.goal_pos}

    def reset(self, seed: int = None) -> dict[str, tuple[int, int]]:
        super().reset(seed=seed)
        self.agent_pos = self.start_pos
        return self._get_observation()

    def step(self, action: int) -> tuple[dict[str, tuple[int, int]]:, float, bool, dict]:
        direction = self.action_to_direction_map[action]
        if action == 0:  # North
            rand = self.np_random.random()
            if rand < (self.noise / 2):
                direction = (0, -1)  # randomly goes West
            elif rand < self.noise:
                direction = (0, 1)  # randomly goes East

        # update position and clip ensure it remains within grid bounds
        next_agent_pos = np.clip(
            np.array(self.agent_pos) + np.array(direction),
            a_min=(0, 0),
            a_max=(self.num_rows - 1, self.num_cols - 1),
        )
        next_agent_pos = tuple(next_agent_pos)

        # Check if hit obstacle
        if next_agent_pos == self.obstacle_pos:
            print("Ran into an obstacle!")
            return self._get_observation(), self.living_reward, False, {}

        # Update agent state
        self.agent_pos = next_agent_pos

        # Check if goal reached
        if next_agent_pos == self.goal_pos:
            print("Goal reached!")
            return self._get_observation(), self.goal_reward, True, {}

        # Check if terminal pos reached
        if next_agent_pos == self.terminal_pos:
            print("Uh oh, terminal reached!")
            return self._get_observation(), self.terminal_reward, True, {}

        return self._get_observation(), self.living_reward, False, {}

    def render(self) -> None:
        desc = self.rendering_grid.tolist()
        desc = [[c.decode("utf-8") for c in line] for line in desc]
        row, col = self.agent_pos
        desc[row][col] = "X"
        desc[row][col] = utils.colorize(desc[row][col], "green", highlight=True)
        print("\n".join("".join(row) for row in desc) + "\n")


if __name__ == "__main__":
    env = GridWorld()
    done = False
    state = env.reset(1)
    step = 0
    total_rewards = 0
    while not done:
        step += 1
        action = env.np_random.integers(0, 4)
        env.render()
        next_state, reward, done, info = env.step(action)
        print(
            f"{state} + {env.action_to_direction_map[action]} --> {next_state} || reward={reward} || done={done}"
        )
        env.render()
        print("----------------------------------------------")
        state = next_state
        total_rewards += reward

    print(f"Num steps: {step} -- rewards: {total_rewards}")
