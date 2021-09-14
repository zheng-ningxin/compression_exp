import time
import numpy as np
import torch
import random
import os
import nni
from nni.algorithms.compression.pytorch.pruning import L1FilterPruner, LevelPruner
from nni.compression.pytorch import ModelSpeedup, apply_compression_results
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
    torch.cuda.manual_seed(seed) # Sets the seed for generating random numbers for the current GPU. It’s safe to call this function if CUDA is not available; in that case, it is silently ignored.
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True

def contruct_model_from_ckpt(model, dummy_input, ckpt_path):
    assert os.path.exists(ckpt_path)
    ckpt = torch.load(ckpt_path)

    cfg_list = []
    for name, module in model.named_modules():
        if isinstance(module, (torch.nn.Conv2d)):
            weight = ckpt[name+'.weight']
            sparsity = 1.0 - (weight.size(0)-0.1)/module.weight.size(0)
            if sparsity> 0 and sparsity<1:
                cfg_list.append({'op_names':[name], 'op_types':['Conv2d', 'Linear'], 'sparsity':sparsity})

    pruner = L1FilterPruner(model, cfg_list, dependency_aware=True, dummy_input=dummy_input)
    pruner.compress()
    pruner.export_model('./tmp_weight', './tmp_mask')
    pruner._unwrap_model()
    ms = ModelSpeedup(model, dummy_input, './tmp_mask')
    ms.speedup_model()
    model.load_state_dict(ckpt)
    return model
