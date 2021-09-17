import tensorflow as tf
from typing import Tuple, Union


class BaseAgent:
    def run_episode(self):
        raise NotImplementedError

    def run_agent(self):
        raise NotImplementedError

    def save_models(self) -> None:
        raise NotImplementedError

    def load_models(self) -> Union[tf.keras.Model, Tuple[tf.keras.Model]]:
        raise NotImplementedError



