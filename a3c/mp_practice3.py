import numpy as np
import multiprocessing as mp
# import tensorflow as tf


input_data = np.arange(10)
labels = 2 * input_data + 5

class MyClass:
    def __init__(self):
        self.model = self.create_model()

    def create_model(self):
        import tensorflow as tf

        model = tf.keras.Sequential([
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1)
        ])
        model.compile(optimizer='rmsprop', loss='mse', metrics=['mse'])
        return model

    def predict(self):
        self.model.predict(input_data)


def my_func():
    worker = MyClass()
    worker.predict()

if __name__ == '__main__':
    num = 2

    processes = []
    for i in range(num):
        p = mp.Process(target=my_func)
        processes.append(p)
        # my_func()

    for proc in processes:
        proc.start()
    for proc in processes:
        proc.join()
    for proc in processes:
        proc.terminate()
