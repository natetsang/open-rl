from blob import Blob
import numpy as np
import cv2
import os
import tensorflow as tf
from PIL import Image
import random


class BlobEnv:
    SIZE = 10
    NUM_STEPS_PER_EPISODE = 200
    RETURN_IMAGES = True
    MOVE_PENALTY = 1
    ENEMY_PENALTY = 300
    FOOD_REWARD = 25
    ACTION_SPACE_SIZE = 9

    # This is for the rendered env. The "3" is because we have RGB values.
    OBSERVATION_SPACE_VALUES = (SIZE, SIZE, 3)

    # Plotting colors
    PLAYER_COLOR = (255, 175, 0)
    FOOD_COLOR = (0, 255, 0)
    ENEMY_COLOR = (0, 0, 255)

    def reset(self):
        self.player = Blob(self.SIZE)
        self.food = Blob(self.SIZE)
        self.enemy = Blob(self.SIZE)

        # Make sure food and player are instantiated on the same tile
        while self.food == self.player:
            self.food = Blob(self.SIZE)

        # Make sure enemy, player, and food are on different tiles.
        while self.enemy == self.player or self.enemy == self.food:
            self.enemy = Blob(self.SIZE)

        self.episode_step = 0

        if self.RETURN_IMAGES:
            observation = np.array(self.get_image())
        else:
            observation = (self.player-self.food) + (self.player-self.enemy)
        return observation

    def step(self, action):
        self.episode_step += 1
        self.player.action(action)

        #### MAYBE ###
        #self.enemy.move()
        #self.food.move()
        ##############

        if self.RETURN_IMAGES:
            new_observation = np.array(self.get_image())
        else:
            new_observation = (self.player-self.food) + (self.player-self.enemy)

        # Determine reward
        if self.player == self.enemy:
            reward = -self.ENEMY_PENALTY
        elif self.player == self.food:
            reward = self.FOOD_REWARD
        else:
            reward = -self.MOVE_PENALTY

        if (reward == self.FOOD_REWARD) or (reward == -self.ENEMY_PENALTY) or (self.episode_step >= self.NUM_STEPS_PER_EPISODE):
            done = True
        else:
            done = False

        return new_observation, reward, done

    def render(self):
        img = self.get_image()
        img = img.resize((300, 300))  # resizing so we can see our agent in all its glory.
        cv2.imshow("image", np.array(img))  # show it!
        cv2.waitKey(1)

    """
    For CNN. Ultimately, in this example we are fitting the model based on image data. Thus, given the state of the game
    we need to construct an image. So we are initializing the grid as having
    (0, 0, 0) RGB for each tile. Then we color the player, enemy, and food tiles.

    Then we return this image, so we can train on IMAGE data.
    """
    def get_image(self):
        env = np.zeros((self.SIZE, self.SIZE, 3), dtype=np.uint8)  # starts an rbg of our size
        env[self.food.x][self.food.y] = self.FOOD_COLOR  # sets the food location tile to green color
        env[self.enemy.x][self.enemy.y] = self.ENEMY_COLOR  # sets the enemy location to red
        env[self.player.x][self.player.y] = self.PLAYER_COLOR  # sets the player tile to blue
        img = Image.fromarray(env, 'RGB')  # reading to rgb. Apparently. Even tho color definitions are bgr. ???
        return img
