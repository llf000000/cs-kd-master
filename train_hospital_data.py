from __future__ import print_function

import random
import numpy as np
import torch
import os
def seed_torch(seed=42):
    random.seed(seed) # Python的随机性
    os.environ["PYTHONHASHSEED"] = str(seed) # 设置python哈希种子，为了禁止hash随机化，使得实验可复现
    np.random.seed(seed) # numpy的随机性
    torch.manual_seed(seed) # torch的CPu随机性，为CPU设置随机种子
    torch.cuda.manual_seed(seed) # torch的GPU随机性，为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPu.torch的GPu随机性，为所有GPU设置随机种子
    torch.backends.cudnn.benchmark = False # if benchmark=True, deterministic will be Falsetorch.backends.cudnn.deterministic = True # 选择确定性算法
    torch.backends.cudnn.deterministic = True # 选择确定性算法
seed=42
seed_torch(seed)

import argparse
import csv
import os, logging

import numpy as np
import torch
from torch.autograd import Variable, grad
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms as transforms

import models
from utils import progress_bar, set_logging_defaults
from datasets import load_dataset

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
parser = argparse.ArgumentParser(description='CS-KD Training')
parser.add_argument('--lr', default=0.1, type=float, help='learning rate')
parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
parser.add_argument('--model', default="CIFAR_ResNet18", type=str, help='model type (32x32: CIFAR_ResNet18, CIFAR_DenseNet121, 224x224: resnet18, densenet121)') # 模型类型
parser.add_argument('--name', default='0', type=str, help='name of run') # 运行名称，默认为'0'。这可以用来标识不同的训练运行
parser.add_argument('--batch-size', default=4, type=int, help='batch size')
parser.add_argument('--epoch', default=200, type=int, help='total epochs to run')
parser.add_argument('--decay', default=1e-4, type=float, help='weight decay') # 权重衰减
parser.add_argument('--ngpu', default=1, type=int, help='number of gpu')
parser.add_argument('--sgpu', default=0, type=int, help='gpu index (start)')
parser.add_argument('--dataset', default='hospital_data', type=str, help='the name for dataset cifar100 | tinyimagenet | CUB200 | STANFORD120 | MIT67') # 数据集名称
parser.add_argument('--dataroot', default='../hospital_data/', type=str, help='data directory') # 数据目录
parser.add_argument('--saveroot', default='./results', type=str, help='save directory')
parser.add_argument('--cls', '-cls', action='store_true', help='adding cls loss') # 是否添加分类损失
parser.add_argument('--temp', default=4.0, type=float, help='temperature scaling')
parser.add_argument('--lamda', default=1.0, type=float, help='cls loss weight ratio') # 分类损失权重比

args = parser.parse_args()
use_cuda = torch.cuda.is_available()

best_val = 0  # best validation accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

cudnn.benchmark = True

# Data
print('==> Preparing dataset: {}'.format(args.dataset))
if not args.cls:
    trainloader, valloader = load_dataset(args.dataset, args.dataroot, batch_size=args.batch_size)
else:
    trainloader, valloader = load_dataset(args.dataset, args.dataroot, 'pair', batch_size=args.batch_size)


num_class = trainloader.dataset.num_classes

print('Number of train dataset: ' ,len(trainloader.dataset))
print('Number of validation dataset: ' ,len(valloader.dataset))

# Model
print('==> Building model: {}'.format(args.model))

net = models.load_model(args.model, num_class)
# print(net)

if use_cuda:
    torch.cuda.set_device(args.sgpu)
    net.cuda()
    print(torch.cuda.device_count())
    print('Using CUDA..')

if args.ngpu > 1:
    net = torch.nn.DataParallel(net, device_ids=list(range(args.sgpu, args.sgpu + args.ngpu)))

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=0.9, weight_decay=args.decay)

logdir = os.path.join(args.saveroot, args.dataset, args.model, args.name)
set_logging_defaults(logdir, args)
logger = logging.getLogger('main')
logname = os.path.join(logdir, 'log.csv')


# Resume
if args.resume: # 从检查点恢复训练
    # Load checkpoint.
    print('==> Resuming from checkpoint..')
    checkpoint = torch.load(os.path.join(logdir, 'ckpt.t7')) 
    net.load_state_dict(checkpoint['net']) # 模型状态
    optimizer.load_state_dict(checkpoint['optimizer']) # 优化器状态
    best_acc = checkpoint['acc'] # 最佳验证准确率 
    start_epoch = checkpoint['epoch'] + 1 # 开始训练周期
    rng_state = checkpoint['rng_state'] # 随机数生成器的内部状态
    torch.set_rng_state(rng_state)

criterion = nn.CrossEntropyLoss()


class KDLoss(nn.Module):
    def __init__(self, temp_factor):
        super(KDLoss, self).__init__()
        self.temp_factor = temp_factor
        self.kl_div = nn.KLDivLoss(reduction="sum")

    def forward(self, input, target):
        log_p = torch.log_softmax(input/self.temp_factor, dim=1)
        q = torch.softmax(target/self.temp_factor, dim=1)
        loss = self.kl_div(log_p, q)*(self.temp_factor**2)/input.size(0)
        return loss

kdloss = KDLoss(args.temp)

def train(epoch):
    print('\nEpoch: %d' % epoch)
    net.train()
    train_loss = 0
    correct = 0
    total = 0
    train_cls_loss = 0
    for batch_idx, (inputs, targets) in enumerate(trainloader):
        if use_cuda:
            inputs, targets = inputs.cuda(), targets.cuda()

        batch_size = inputs.size(0)

        if not args.cls: # 如果未启用知识蒸馏
            outputs = net(inputs)
            loss = torch.mean(criterion(outputs, targets)) # 交叉熵损失
            train_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).sum().float().cpu()
        else:
            '''代码首先处理前面一半的批次数据'''
            targets_ = targets[:batch_size//2] # targets_变量存储了用于上半部分数据的目标标签。
            outputs = net(inputs[:batch_size//2]) # 
            loss = torch.mean(criterion(outputs, targets_)) # 交叉熵损失
            train_loss += loss.item()

            '''剩下一半的批次数据'''
            with torch.no_grad(): # 在不计算梯度的情况下
                outputs_cls = net(inputs[batch_size//2:]) # 剩下一半的数据
            
            cls_loss = kdloss(outputs, outputs_cls.detach()) # 知识蒸馏损失 利用模型自身的输出作为额外的指导信息，通过正则化的方式促进魔心搞学习更加鲁棒的特征表示，以提高模型的泛化能力
            '''
            系数args.lamda: 这个系数控制知识蒸馏损失相对于主损失（在这个场景下是交叉熵损失）的重要性。
            通过调整args.lamda的值，可以平衡模型在准确度和泛化之间的权衡。较大的args.lamda值意味着给
            予知识蒸馏损失更高的重视，这可以被视为加强了正则化效果，有助于模型在未见过的数据上表现得更好，
            但也可能会牺牲一些训练集上的性能。
            '''
            loss += args.lamda * cls_loss
            train_cls_loss += cls_loss.item()

            _, predicted = torch.max(outputs, 1)
            total += targets_.size(0)
            correct += predicted.eq(targets_.data).sum().float().cpu()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        progress_bar(batch_idx, len(trainloader),
                     'Loss: %.3f | Acc: %.3f%% (%d/%d) | Cls: %.3f '
                     % (train_loss/(batch_idx+1), 100.*correct/total, correct, total, train_cls_loss/(batch_idx+1)))

    logger = logging.getLogger('train')
    logger.info('[Epoch {}] [Loss {:.3f}] [cls {:.3f}] [Acc {:.3f}]'.format(
        epoch,
        train_loss/(batch_idx+1),
        train_cls_loss/(batch_idx+1),
        100.*correct/total))

    return train_loss/batch_idx, 100.*correct/total, train_cls_loss/batch_idx

def val(epoch):
    global best_val
    net.eval()
    val_loss = 0.0
    correct = 0.0
    total = 0.0

    # Define a data loader for evaluating
    loader = valloader

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(loader):
            if use_cuda:
                inputs, targets = inputs.cuda(), targets.cuda()

            outputs = net(inputs)
            loss = torch.mean(criterion(outputs, targets))

            val_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum().float()

            progress_bar(batch_idx, len(loader),
                         'Loss: %.3f | Acc: %.3f%% (%d/%d) '
                         % (val_loss/(batch_idx+1), 100.*correct/total, correct, total))

    acc = 100.*correct/total
    logger = logging.getLogger('val')
    logger.info('[Epoch {}] [Loss {:.3f}] [Acc {:.3f}]'.format(
        epoch,
        val_loss/(batch_idx+1),
        acc))

    if acc > best_val:
        best_val = acc
        checkpoint(acc, epoch)

    return (val_loss/(batch_idx+1), acc)


def checkpoint(acc, epoch):
    # Save checkpoint.
    print('Saving..')
    state = {
        'net': net.state_dict(),
        'optimizer': optimizer.state_dict(),
        'acc': acc,
        'epoch': epoch,
        'rng_state': torch.get_rng_state()
    }
    torch.save(state, os.path.join(logdir, 'ckpt.t7'))


def adjust_learning_rate(optimizer, epoch):
    """decrease the learning rate at 100 and 150 epoch"""
    lr = args.lr
    if epoch >= 0.5 * args.epoch:
        lr /= 10
    if epoch >= 0.75 * args.epoch:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# Logs
for epoch in range(start_epoch, args.epoch):
    train_loss, train_acc, train_cls_loss = train(epoch)
    val_loss, val_acc = val(epoch)
    adjust_learning_rate(optimizer, epoch)

print("Best Accuracy : {}".format(best_val))
logger = logging.getLogger('best')
logger.info('[Acc {:.3f}]'.format(best_val))
