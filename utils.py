import time
import numpy as np
import torch
import random



def measure_time(model, data, runtimes=1000):
    model.eval()
    times = []
    with torch.no_grad():
        for runtime in range(runtimes):
            start = time.time()
            torch.cuda.synchronize()
            model(data)
            torch.cuda.synchronize()
            end = time.time()
            times.append(end-start)
    _drop = int(runtimes * 0.1)
    mean = np.mean(times[_drop:-1*_drop])
    std = np.std(times[_drop:-1*_drop])
    return mean, std

def init_seed(seed):
    torch.manual_seed(seed)
    random.seed(seed)