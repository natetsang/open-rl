import tensorflow as tf


class BaseAgent:
    def test_agent(self):
        raise NotImplementedError

    def run_episode(self):
        raise NotImplementedError

    def save_models(self) -> None:
        raise NotImplementedError

    def load_models(self) -> tf.keras.Model:
        raise NotImplementedError



