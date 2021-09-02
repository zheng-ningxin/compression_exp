# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.
'''
Compression Experiments codes
'''

import argparse
import random
import time
from types import new_class
from nni.compression.pytorch.speedup.compressor import ModelSpeedup
import numpy as np
import os
import json
import torch
from torch.jit import Error
import torchvision
from torch.optim.lr_scheduler import StepLR, MultiStepLR, CosineAnnealingLR
from torchvision import datasets, transforms
from nni.algorithms.compression.pytorch.pruning import L1FilterPruner, L2FilterPruner, FPGMPruner
from nni.algorithms.compression.pytorch.pruning import ADMMPruner, NetAdaptPruner, AMCPruner
from nni.compression.pytorch.utils.counter import count_flops_params
from pruners.simulated_annealing_pruner import SimulatedAnnealingPruner
from pruners.auto_compress_pruner import AutoCompressPruner
# import the cifar models
from models.mnist.lenet import LeNet as CifarLeNet
from models.cifar10.vgg import VGG as CifarVGG
from models.cifar10.mobilenetv2 import MobileNetV2 as CifarMobileNetV2
from models.cifar10.resnet import ResNet18 as CifarResNet18
from models.cifar10.resnet import ResNet50 as CifarResNet50
# import the imagenet models
from models.mobilenet import MobileNet as ImageNetMobileNet
from utils import *

cifar_models = {'lenet': CifarLeNet, 'vgg': CifarVGG, 'mobilenetv2': CifarMobileNetV2,
                'resnet18': CifarResNet18, 'resnet50': CifarResNet50}
imagenet_models = {'mobilenet': ImageNetMobileNet, 'vgg11': torchvision.models.vgg11}
mnist_models = {}
model_args = {'imagenet':{'mobilenet':[1000]}}

def get_data(dataset, data_dir, batch_size):
    '''
    get data
    '''
    kwargs = {'num_workers': 16, 'pin_memory': True} if torch.cuda.is_available() else {
    }

    if dataset == 'mnist':
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST(data_dir, train=True, download=True,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        val_loader = torch.utils.data.DataLoader(
            datasets.MNIST(data_dir, train=False,
                           transform=transforms.Compose([
                               transforms.ToTensor(),
                               transforms.Normalize((0.1307,), (0.3081,))
                           ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        criterion = torch.nn.NLLLoss()
    elif dataset == 'cifar10':
        normalize = transforms.Normalize(
            (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(data_dir, train=True, transform=transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]), download=True),
            batch_size=batch_size, shuffle=True, **kwargs)

        val_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(data_dir, train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                normalize,
            ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        criterion = torch.nn.CrossEntropyLoss()
    elif dataset == 'imagenet':
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        train_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(os.path.join(data_dir, 'train'),
                                 transform=transforms.Compose([
                                     transforms.RandomResizedCrop(224),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     normalize,
                                 ])),
            batch_size=batch_size, shuffle=True, **kwargs)

        val_loader = torch.utils.data.DataLoader(
            datasets.ImageFolder(os.path.join(data_dir, 'val'),
                                 transform=transforms.Compose([
                                     transforms.Resize(256),
                                     transforms.CenterCrop(224),
                                     transforms.ToTensor(),
                                     normalize,
                                 ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        criterion = torch.nn.CrossEntropyLoss()

    return train_loader, val_loader, criterion


def train(model, device, train_loader, criterion, optimizer, callback=None):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        # callback should be inserted between loss.backward() and optimizer.step()
        if callback:
            callback()
        optimizer.step()
        if batch_idx % 20 == 0:
            print('Training[{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, criterion, val_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            # sum up batch loss
            test_loss += criterion(output, target).item()
            # get the index of the max log-probability
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(val_loader.dataset)
    accuracy = correct / len(val_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(val_loader.dataset), 100. * accuracy))

    return accuracy


def get_model(args):
    models = None
    tmp_args = []
    if args.dataset in model_args and args.model in model_args[args.dataset]:
        tmp_args = model_args[args.dataset][args.model]
    if args.dataset == 'cifar10':
        models = cifar_models        
    elif args.dataset == 'imagenet':
        models = imagenet_models
    elif args.dataset == 'mnist':
        models = mnist_models
    else:
        raise Exception('Dataset not supported')
    model = models[args.model](*tmp_args)
    if args.load_pretrained_model is not None:
        model.load_state_dict(torch.load(args.load_pretrained_model))

    return model


def get_dummy_input(args, device):
    if args.dataset == 'mnist':
        dummy_input = torch.randn([args.batch_size, 1, 28, 28]).to(device)
    elif args.dataset == 'cifar10':
        dummy_input = torch.randn([args.batch_size, 3, 32, 32]).to(device)
    elif args.dataset == 'imagenet':
        dummy_input = torch.randn(
            [args.batch_size, 3, 224, 224]).to(device)
    return dummy_input


def get_input_size(dataset):
    if dataset == 'mnist':
        input_size = (1, 1, 28, 28)
    elif dataset in ['cifar10']:
        input_size = (1, 3, 32, 32)
    elif dataset == 'imagenet':
        input_size = (1, 3, 224, 224)
    return input_size


def main(args):
    # prepare dataset
    # torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader, val_loader, criterion = get_data(
        args.dataset, args.data_dir, args.batch_size)
    model = get_model(args).to(device)
    ori_acc = test(model, device, criterion, val_loader)
    print('Original Accuracy before the pruning', ori_acc)

    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_short_finetune, momentum=0.9, weight_decay=5e-4)
    def short_term_fine_tuner(model, epochs=args.short_term_finetune):
        for epoch in range(epochs):
            print('Short term finetune epoch :', epoch)
            train(args, model, device, train_loader,
                  criterion, optimizer, epoch)

    def trainer(model, optimizer, criterion, epoch, callback):
        return train(args, model, device, train_loader, criterion, optimizer, epoch=epoch, callback=callback)

    def evaluator(model):
        return test(model, device, criterion, val_loader)
    dummy_input = get_dummy_input(args, device)
    # used to save the performance of the original & pruned & finetuned models
    result = {'flops': {}, 'params': {},
              'performance': {}, 'time_mean': {}, 'time_std': {}}
    # measure the original performance before pruning
    flops, params, _ = count_flops_params(model, get_input_size(args.dataset))
    ori_time_mean, ori_time_std = measure_time(model, dummy_input)
    result['flops']['original'] = flops
    result['params']['original'] = params
    result['time_mean']['original'] = ori_time_mean
    result['time_std']['original'] = ori_time_std
    evaluation_result = evaluator(model)
    print('Evaluation result (original model): %s' % evaluation_result)
    result['performance']['original'] = evaluation_result

    ## Generate the config list for NNI pruners
    # module types to prune, only "Conv2d" supported for channel pruning
    if args.base_algo in ['l1', 'l2']:
        op_types = ['Conv2d']
    elif args.base_algo == 'level':
        op_types = ['default']

    config_list = [{
        'sparsity': args.sparsity,
        'op_types': op_types
    }]

    if args.sparsity_config:
        config_list = json.load(args.sparsity_config)

    if args.pruner == 'L1FilterPruner':
        pruner = L1FilterPruner(
            model, config_list, dependency_aware=args.dependency, dummy_input=dummy_input)
    elif args.pruner == 'L2FilterPruner':
        pruner = L2FilterPruner(
            model, config_list, dependency_aware=args.dependency, dummy_input=dummy_input)
    elif args.pruner == 'FPGMPruner':
        pruner = FPGMPruner(
            model, config_list, dependency_aware=args.dependency, dummy_input=dummy_input)
    elif args.pruner == 'NetAdaptPruner':
        pruner = NetAdaptPruner(model, config_list, short_term_fine_tuner=short_term_fine_tuner, evaluator=evaluator,
                                base_algo=args.base_algo, experiment_data_dir=args.experiment_data_dir)
    elif args.pruner == 'AMCPruner':
        def amc_evaluator(val_loader, model):
            return test(model, device, criterion, val_loader)
        pruner = AMCPruner(model, config_list, amc_evaluator,
                           val_loader, flops_ratio=args.sparsity)
    elif args.pruner == 'ADMMPruner':
        # users are free to change the config here
        if args.model == 'LeNet':
            if args.base_algo in ['l1', 'l2']:
                config_list = [{
                    'sparsity': 0.8,
                    'op_types': ['Conv2d'],
                    'op_names': ['conv1']
                }, {
                    'sparsity': 0.92,
                    'op_types': ['Conv2d'],
                    'op_names': ['conv2']
                }]
            elif args.base_algo == 'level':
                config_list = [{
                    'sparsity': 0.8,
                    'op_names': ['conv1']
                }, {
                    'sparsity': 0.92,
                    'op_names': ['conv2']
                }, {
                    'sparsity': 0.991,
                    'op_names': ['fc1']
                }, {
                    'sparsity': 0.93,
                    'op_names': ['fc2']
                }]
        else:
            raise ValueError('Example only implemented for LeNet.')
        pruner = ADMMPruner(model, config_list, trainer=trainer,
                            num_iterations=2, training_epochs=2)
    elif args.pruner == 'SimulatedAnnealingPruner':
        pruner = SimulatedAnnealingPruner(
            model, config_list, evaluator=evaluator,  base_algo=args.base_algo,
            cool_down_rate=args.cool_down_rate, experiment_data_dir=args.experiment_data_dir)
    elif args.pruner == 'AutoCompressPruner':
        pruner = AutoCompressPruner(
            model, config_list, trainer=trainer, evaluator=evaluator,
            num_iterations=3, optimize_mode='maximize', base_algo=args.base_algo,
            cool_down_rate=args.cool_down_rate, admm_num_iterations=5, admm_training_epochs=5,
            experiment_data_dir=args.experiment_data_dir)
    else:
        raise ValueError(
            "Pruner not supported.")

    # Pruner.compress() returns the masked model
    # but for AutoCompressPruner, Pruner.compress() returns directly the pruned model
    model = pruner.compress()
    evaluation_result = evaluator(model)
    print('Evaluation result (masked model): %s' % evaluation_result)
    result['performance']['pruned'] = evaluation_result
    weight_path = os.path.join(args.experiment_data_dir, 'model_masked.pth')
    mask_path = os.path.join(args.experiment_data_dir, 'mask.pth')
    pruner.export_model(
        weight_path, mask_path)
    print('Masked model saved to %s', args.experiment_data_dir)

    pruner._unwrap_model()

    ms = ModelSpeedup(model, dummy_input, mask_path)
    ms.speedup_model()
    # create a new optimizer for speedup model
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr_finetune, momentum=0.9, weight_decay=5e-4)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.fine_tune_epochs)
    best_acc = 0
    for epoch in range(args.fine_tune_epochs):
        train(model, device, train_loader, criterion, optimizer)
        scheduler.step()
        acc = evaluator(model)
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), os.path.join(args.experiment_data_dir, 'model_fine_tuned.pth'))

    evaluation_result = evaluator(model)
    print('Evaluation result (speed up model): %s' % evaluation_result)
    result['performance']['speedup'] = evaluation_result
    torch.save(model.state_dict(), os.path.join(args.experiment_data_dir, 'speedup_model.pth'))

    with open(os.path.join(args.experiment_data_dir, 'result.json'), 'w+') as f:
        json.dump(result, f)


if __name__ == '__main__':
    def str2bool(s):
        if isinstance(s, bool):
            return s
        if s.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        if s.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser(
        description='PyTorch Example for SimulatedAnnealingPruner')

    # dataset and model
    parser.add_argument('--dataset', type=str, default='imagenet',
                        help='dataset to use, mnist, cifar10 or imagenet')
    parser.add_argument('--data-dir', type=str, default='./data/',
                        help='dataset directory')
    parser.add_argument('--model', type=str, default='vgg16',
                        help='model to use, LeNet, vgg16, resnet18 or resnet50')
    parser.add_argument('--load-pretrained-model', type=str, default=None,
                        help='Path to load the checkpoint')

    parser.add_argument('--batch-size', type=int, default=64,
                        help='input batch size for training (default: 64)')

    
    parser.add_argument('--fine-tune-epochs', type=int, default=5,
                        help='epochs to fine tune')
    parser.add_argument('--experiment-data-dir', type=str, default='./experiment_data',
                        help='For saving experiment data')
    parser.add_argument('--sparsity_config', type=str, default=None,
                        help='the path of the sparsity config file')
    # pruner
    parser.add_argument('--pruner', type=str, default='SimulatedAnnealingPruner',
                        help='pruner to use')
    parser.add_argument('--base-algo', type=str, default='l1',
                        help='base pruning algorithm. level, l1 or l2')
    parser.add_argument('--sparsity', type=float, default=0.1,
                        help='target overall target sparsity')
    # param for SimulatedAnnealingPruner
    parser.add_argument('--cool-down-rate', type=float, default=0.97,
                        help='cool down rate')
    # param for NetAdaptPruner
    parser.add_argument('--sparsity-per-iteration', type=float, default=0.05,
                        help='sparsity_per_iteration of NetAdaptPruner')

    parser.add_argument('--dependency', type=str2bool, default=False,
                        help='if enable the constraint-aware pruner')
    parser.add_argument('--lr_finetune', type=float, default=0.01,
                        help='The learning rate for the finetuning')
    parser.add_argument('--lr_short_finetune', type=float, default=0.001,
                        help='The learning rate for the finetuning')
    parser.add_argument('--short_term_finetune', type=int,
                        default=20, help='the short term finetune epochs')
 
    parser.add_argument('--aligned', default=8, type=int,
                        help='The number of the pruned filter should be aligned with')
    parser.add_argument('--seed', default=2020, type=int,
                        help='The random seed for torch and random module.')
    args = parser.parse_args()
    # random init the seed
    init_seed(args.seed)

    if not os.path.exists(args.experiment_data_dir):
        os.makedirs(args.experiment_data_dir)

    main(args)
