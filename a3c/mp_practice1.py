import numpy as np
import multiprocessing as mp


def square(i, x, queue):
    print(f"in process {i}")
    queue.put(np.square(x))

if __name__ == '__main__':
    num = mp.cpu_count()

    processes = []
    queue = mp.Queue()
    x = np.arange(64)
    for i in range(num):
        start_index = num*i
        p = mp.Process(target=square, args=(i, x[start_index:start_index+num], queue))
        processes.append(p)

    for proc in processes:
        proc.start()

    for proc in processes:
        proc.join()

    for proc in processes:
        proc.terminate()

    results = []
    while not queue.empty():
        results.append(queue.get())

    print(results)
