import torch
from models.cifar10.mobilenet import MobileNet
from nni.algorithms.compression.pytorch.pruning import L1FilterPruner, L2FilterPruner, FPGMPruner
from nni.compression.pytorch import ModelSpeedup
from nni.compression.pytorch.utils.counter import count_flops_params
from utils import measure_time

model1 = MobileNet().cuda()
data = torch.rand(128, 3, 32, 32).cuda()
t_ori = measure_time(model1, data)
print(t_ori)
model2 = MobileNet().cuda()
cfg_list = [{'op_types':['Conv2d'], 'sparsity':0.1}]
pruner = L1FilterPruner(model2, cfg_list)
pruner.compress()
pruner.export_model('./weight', './mask')
pruner._unwrap_model()
ms = ModelSpeedup(model2, data, './mask')

ms.speedup_model()

t_pruned = measure_time(model2, data)
import pdb; pdb.set_trace()

print('Original:', t_ori)
print('Pruned', t_pruned)

