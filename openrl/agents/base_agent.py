import tensorflow as tf
from typing import Tuple, Union


class BaseAgent:
    def train_episode(self) -> dict:
        """
        Run 1 training episode
        :return: a dictionary of training logs
        """
        raise NotImplementedError

    def run_agent(self) -> Tuple[float, int]:
        """
        Run the agent through one episode in the env to evaluate performance.
        :return: a Tuple where the first value is the episode rewards,
        and the second value is the number of steps before reaching 'done'
        """
        raise NotImplementedError

    def save_models(self) -> None:
        """
        Save the underlying models. This could be a single model (i.e. actor) or it could be multiple.
        :return:
        """
        raise NotImplementedError

    def load_models(self) -> Union[tf.keras.Model, Tuple[tf.keras.Model]]:
        """
        Load the models from the 'save_dir' path, set them to the instance variables,
        and return. This could be a single model or could be at Tuple of models.
        :return: a single or tuple of models
        """
        raise NotImplementedError



