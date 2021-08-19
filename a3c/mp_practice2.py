import numpy as np
import multiprocessing as mp


class MyClass:
    def __init__(self, x):
        self.x = x

    def square(self):
        print("original x: ", self.x)
        self.x = np.square(self.x)
        print("squared x: ", self.x)

def my_func(class_instance):
    class_instance.square()

if __name__ == '__main__':
    num = 2

    processes = []
    workers = []
    x = np.arange(5)
    for i in range(num):
        worker = MyClass(x)
        p = mp.Process(target=my_func, args=(worker,))
        processes.append(p)
        workers.append(worker)


    for proc in processes:
        proc.start()
    for proc in processes:
        proc.join()
    for proc in processes:
        proc.terminate()
