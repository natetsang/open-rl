import gym
import numpy as np
from gym import spaces, utils


class GridWorld(gym.Env):
    def __init__(self, noise: float = 0.2, living_reward: float = -0.1):
        self.num_rows = 3
        self.num_cols = 4
        self.actions_map = {0: "N", 1: "S", 2: "W", 3: "E"}
        self.action_space = spaces.Discrete(len(self.actions_map))

        self.observation_space = spaces.Discrete(1)  # the states are enumerated
        self.noise = noise

        # Rewards
        self.living_reward = living_reward
        self.goal_reward = 1
        self.terminal_reward = -1

        self.start_pos = (2, 0)
        self.goal_pos = (0, 3)
        self.terminal_pos = (1, 3)
        self.obstacle_pos = (1, 1)

        self.index_to_coordinate_map = {
            # row 1
            0: (0, 0),
            1: (0, 1),
            2: (0, 2),
            3: (0, 3),  # Goal state
            # row 2
            4: (1, 0),
            5: (1, 1),  # Wall
            6: (1, 2),
            7: (1, 3),  # Terminal state
            # row 3
            8: (2, 0),  # Start
            9: (2, 1),
            10: (2, 2),
            11: (2, 3),
        }

        self.coordinate_to_index_map = {
            v: k for (k, v) in self.index_to_coordinate_map.items()
        }
        self.state = self.coordinate_to_index_map[self.start_pos]

        # Rendering
        self.rendering_grid = np.asarray(
            [
                "OOOG",
                "OWOT",
                "OOOO",
            ],
            dtype="c",
        )

    def reset(self, seed: int = None) -> int:
        super().reset(seed=seed)
        self.state = self.coordinate_to_index_map[self.start_pos]
        return self.state

    def step(self, action: int) -> tuple[int, float, bool, dict]:
        action_desc = self.actions_map[action]
        state_pos = self.index_to_coordinate_map[self.state]
        if action_desc == "N":
            rand = self.np_random.random()
            if rand < (self.noise / 2):
                # randomly goes West
                action_desc == "W"
            elif rand < self.noise:
                # randomly goes East
                action_desc == "E"
            else:
                # agent goes North as expected
                next_state_pos = (max(0, state_pos[0] - 1), state_pos[1])
        if action_desc == "S":
            next_state_pos = (min(self.num_rows - 1, state_pos[0] + 1), state_pos[1])
        if action_desc == "W":
            next_state_pos = (state_pos[0], max(0, state_pos[1] - 1))
        if action_desc == "E":
            next_state_pos = (state_pos[0], min(self.num_cols - 1, state_pos[1] + 1))

        # Check if hit obstacle
        if next_state_pos == self.obstacle_pos:
            print("Ran into an obstacle!")
            return self.state, self.living_reward, False, {}

        # Update agent state
        self.state = self.coordinate_to_index_map[next_state_pos]

        # Check if goal reached
        if next_state_pos == self.goal_pos:
            print("Goal reached!")
            return self.state, self.goal_reward, True, {}

        # Check if terminal pos reached
        if next_state_pos == self.terminal_pos:
            print("Uh oh, terminal reached!")
            return self.state, self.terminal_reward, True, {}

        return self.state, self.living_reward, False, {}

    def render(self) -> None:
        desc = self.rendering_grid.tolist()
        desc = [[c.decode("utf-8") for c in line] for line in desc]
        row, col = self.index_to_coordinate_map[self.state]

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
            f"{env.index_to_coordinate_map[state]} + {env.actions_map[action]} --> "
            + f"{env.index_to_coordinate_map[next_state]} || reward={reward} || done={done}"
        )
        env.render()
        print("----------------------------------------------")
        state = next_state
        total_rewards += reward

    print(f"Num steps: {step} -- rewards: {total_rewards}")
