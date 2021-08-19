"""
In this script, we train a q-table on a custom environment.
"""

import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import pickle
from matplotlib import style
import time
style.use("ggplot")

np.random.seed(42)  # this will make us able to compare models

# 10x10 grid
GRID_SIZE = 10

# Game parameters
HM_EPISODES = 25000
MOVE_PENALTY = 1
ENEMY_PENALTY = 300
FOOD_REWARD = 25
epsilon = 0.9
EPS_DECAY = 0.9998
SHOW_EVERY = 3000
STEPS = 200
start_q_table = None  # or filename if you have existing

LEARNING_RATE = 0.1
DISCOUNT = 0.95

# Plotting colors
PLAYER_COLOR = (255, 175, 0)
FOOD_COLOR = (0, 255, 0)
ENEMY_COLOR = (0, 0, 255)

# Observation space would be huge if you had to pass the location of everything
# Instead it will be the relative position of the food and the enemy to the player
# This will reduce the observation space size

class Blob:
    def __init__(self):
        self.x = np.random.randint(0, GRID_SIZE)
        self.y = np.random.randint(0, GRID_SIZE)

    def __str__(self):
        return f"Blob location: ({self.x}, {self.y})"

    def __sub__(self, other_blob):
        """Subtract one blob from another"""
        return (self.x - other_blob.x, self.y - other_blob.y)

    def action(self, choice):
        """It can only move diagonally"""
        if choice == 0:
            self.move(x=1, y=1)
        elif choice == 1:
            self.move(x=-1, y=-1)
        elif choice == 2:
            self.move(x=-1, y=1)
        elif choice == 3:
            self.move(x=1, y=-1)

    def move(self, x=False, y=False):
        if not x:
            self.x += np.random.randint(-1, 2)
        else:
            self.x += x
        if not y:
            self.y += np.random.randint(-1, 2)
        else:
            self.y += y

        # Accounting for the boundary of the game
        # If you move outside the boundary, you get reset to closest tile
        if self.x < 0:
            self.x = 0
        elif self.x > GRID_SIZE - 1:
            self.x = GRID_SIZE - 1
        if self.y < 0:
            self.y = 0
        elif self.y > GRID_SIZE - 1:
            self.y = GRID_SIZE - 1

if start_q_table is None:
    # Create q-table. To do this, we need to randomly populate all possible
    # values in the state observation space. Since
    """
    There are GRID_SIZE * GRID_SIZE entries in the dictionary where
    the key is the unique location of the player and the food, and the value
    is the q_value of each possible action at that unique location.

    q_table =
    {
    ((x1, y1), (x2, y2)): [a1, a2, a3, a4],
    ((x1, y1), (x2, y2)): [a1, a2, a3, a4],
    ...
    }

    The first tuple represents the relative distance between the food and player,
    while the second tuple represents the relative distance between the enemy
    and player.

    These values go from -GRID_SIZE + 1 to GRID_SIZE because we are looking at
    the DELTA between to objects, not the location of the objects themselves!

    Example:
    ((-1, 0), (1, 1)) means that the food is located to the left of the player
    by one tile. The enemy is located to the right of the player by one tile,
    and above the player by one tile.

    Thus to get the q-values for a given state, we would do this:
        q_table([((x1, y1), (x2, y2))]) = q-values for all possible actions
    """
    q_table = dict()
    for x1 in range(-GRID_SIZE + 1, GRID_SIZE):
        for y1 in range(-GRID_SIZE + 1, GRID_SIZE):
            for x2 in range(-GRID_SIZE + 1, GRID_SIZE):
                for y2 in range(-GRID_SIZE + 1, GRID_SIZE):
                    q_table[((x1, y1), (x2, y2))] = [np.random.uniform(-5, 0) for i in range(4)]
else:
    with open(start_q_table, "rb") as f:
        q_table = pickle.load(f)

episode_rewards = []
for episode in range(HM_EPISODES):
    # Initialize blobs at random locations!
    player = Blob()
    food = Blob()
    enemy = Blob()

    # Print results to CLI
    if episode % SHOW_EVERY == 0:
        print(f"Episode # {episode}, Epsilon: {round(epsilon, 3)}")
        print(f"Mean for the last {SHOW_EVERY} episodes, {round(np.mean(episode_rewards[-SHOW_EVERY:]), 3)}")
        show = True
    else:
        show = False

    episode_reward = 0
    for i in range(STEPS):
        # The observation is the relative distance between the player and the food & enemy blobs
        # obs is in the form ((x1, y1), (x2, y2))
        obs = (player - food, player - enemy)

        # Take an action
        if np.random.random() > epsilon:
            # Note that using q-tables, if we've never been to this exact state, we will
            # take an action randomly based on the random initialization of values
            action = np.argmax(q_table[obs])
        else:
            action = np.random.randint(0, 4)
        player.action(action)

        #### maybe later we'll have the enemy or food move too....
        # enemy.move()
        # food.move()
        ####

        # Calculate reward from taking that action
        if player.x == enemy.x and player.y == enemy.y:
            reward = -ENEMY_PENALTY
        elif player.x == food.x and player.y == food.y:
            reward = FOOD_REWARD
        else:
            reward = -MOVE_PENALTY

        # Calculate current q from being in s and taking a
        current_q = q_table[obs][action]

        # Step forward:
        # We already took the action above, so now let's
        # calculate the new observation and max q-value of new state
        new_obs = (player - food, player - enemy)
        # Get the max q value of being in this new state
        max_future_q = np.max(q_table[new_obs])

        # Calculate new Q-value given the current state
        # This is better than the reward update below, which was used in the videos
        new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

        # The video hard codes the reward for reaching na terminal state.
        # Empirically I found it actually it works better to always use Bellman's to update.
        # I've seen examples of both ways.
        # if reward == FOOD_REWARD:
        #     new_q = FOOD_REWARD
        # elif reward == -ENEMY_PENALTY:
        #     new_q = -ENEMY_PENALTY
        # else:
        #     # Slightly alter the value
        #     new_q = (1 - LEARNING_RATE) * current_q + LEARNING_RATE * (reward + DISCOUNT * max_future_q)

        # Update q-table with the q-value we just calculated
        q_table[obs][action] = new_q

        # Show table and get correct colors
        if show:
            env = np.zeros((GRID_SIZE, GRID_SIZE, 3), dtype=np.uint8)
            # in an array, you flip x and y to display in the way we'd think of x and y
            env[food.y][food.x] = FOOD_COLOR
            env[player.y][player.x] = PLAYER_COLOR
            env[enemy.y][enemy.x] = ENEMY_COLOR

            # Actually create the image and display
            img = Image.fromarray(env, "RGB")
            img = img.resize((500, 500))
            cv2.imshow("", np.array(img))
            # If we hit the food or the enemy, then we'll pause
            if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
                if cv2.waitKey(500) & 0xFF == ord("q"):
                    break
            else:
                if cv2.waitKey(20) & 0xFF == ord("q"):
                    break

        # Sum rewards from each step to get total reward for the episode
        episode_reward += reward
        if reward == FOOD_REWARD or reward == -ENEMY_PENALTY:
            # The episode is finished, so stop taking steps!
            break

    episode_rewards.append(episode_reward)
    epsilon *= EPS_DECAY

# Plot
moving_avg = np.convolve(episode_rewards, np.ones((SHOW_EVERY, )) / SHOW_EVERY, mode="valid")

plt.plot([i for i in range(len(moving_avg))], moving_avg)
plt.ylabel(f"reward {SHOW_EVERY} moving avg")
plt.xlabel("episode number")
plt.show()

# Save q-table
with open(f"qtable-{int(time.time())}.pickle", "wb") as f:
    pickle.dump(q_table, f)
