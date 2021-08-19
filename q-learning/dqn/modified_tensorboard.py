import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard


class ModifiedTensorBoard(TensorBoard):
    """
    By default TensorBoard will create a log file every time we do a .fit().
    But we're doing a .fit() like 200 times an episode, for 25000 episodes.
    So it's not realistic we are going to create all of these log files.
    The IO to write to a new file takes a long time, and it takes up a lot of
    space too.
    We just want one log file!
    """

    # Overriding init to set intial step and writer (we want one log file for all .fit() calls)
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)
        self._log_write_dir = self.log_dir  # added this specifically to try to fix error

    # Overriding this method to stop creating default log writer
    def set_model(self, model):
        pass

    # Overrided, saves logs with our step number, otherwise every .fit()
    #will start writing from the 0th step
    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    # Overrided, We train for one batch only, no need to save anything at epoch end
    def on_batch_end(self, batch, logs=None):
        pass

    # Overrided, so won't close writer
    def on_train_end(self, _):
        pass

    # Custom method for saving own metrics
    # Creates writer, writes custom metrics and closes writer
    def update_stats(self, **stats):
        for name, value in stats.items():
            tf.summary.scalar(name=name, data=value, step=self.step)
