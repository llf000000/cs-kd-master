'''
一个loader里前半部分的类别索引和后半部分的类别索引是一样的
比如inputs[:batch_size//2]的index0对应的类别和
inputs[batch_size//2:]的index0对应的类别是一样的
'''
import csv, torchvision, numpy as np, random, os
from PIL import Image

from torch.utils.data import Sampler, Dataset, DataLoader, BatchSampler, SequentialSampler, RandomSampler, Subset
from torchvision import transforms, datasets
from collections import defaultdict


class PairBatchSampler(Sampler):
    '''它是PyTorch的Sampler类的子类，用于在数据加载过程中生成成对样本的批次。'''
    def __init__(self, dataset, batch_size, num_iterations=None):
        self.dataset = dataset
        self.batch_size = batch_size # 每个批次的样本数量。实际生成的批次大小会是这个值的两倍，因为每个样本都将寻找一个配对样本。
        self.num_iterations = num_iterations # 指定迭代的次数。如果未指定，则采样器将基于数据集大小和批次大小自动计算迭代次数。

    def __iter__(self):
        indices = list(range(len(self.dataset))) # 根据num_iterations是否指定，采样器采用不同策略生成每个批次的索引：
        random.shuffle(indices) # 生成整个数据集的索引列表，并对这个列表进行随机化处理。
        for k in range(len(self)):
            if self.num_iterations is None:
                offset = k*self.batch_size
                batch_indices = indices[offset:offset+self.batch_size]
            else: # 如果num_iterations指定，则从数据集中随机选择batch_size个样本作为一个批次，无论数据集的大小。
                batch_indices = random.sample(range(len(self.dataset)),
                                              self.batch_size)

            pair_indices = []
            for idx in batch_indices:
                y = self.dataset.get_class(idx) # 根据其类别（self.dataset.get_class(idx)）从对应类别的样本索引列表中随机选择一个索引，以形成样本对。
                pair_indices.append(random.choice(self.dataset.classwise_indices[y]))

            yield batch_indices + pair_indices # 生成的批次将是原始批次索引和配对索引的合并

    def __len__(self):
        if self.num_iterations is None:
            return (len(self.dataset)+self.batch_size-1) // self.batch_size
        else:
            return self.num_iterations


class DatasetWrapper(Dataset):
    # Additinoal attributes
    # - indices
    # - classwise_indices
    # - num_classes
    # - get_class

    def __init__(self, dataset, indices=None): # dataset: 被封装的基础数据集。
        self.base_dataset = dataset # 保存传入的dataset作为self.base_dataset。
        if indices is None:
            self.indices = list(range(len(dataset))) # 如果未提供indices，则创建一个包含基础数据集中所有索引的列表；
        else:
            self.indices = indices

        # torchvision 0.2.0 compatibility
        if torchvision.__version__.startswith('0.2'):
            if isinstance(self.base_dataset, datasets.ImageFolder):
                self.base_dataset.targets = [s[1] for s in self.base_dataset.imgs]
            else:
                if self.base_dataset.train:
                    self.base_dataset.targets = self.base_dataset.train_labels
                else:
                    self.base_dataset.targets = self.base_dataset.test_labels

        self.classwise_indices = defaultdict(list)
        for i in range(len(self)):
            y = self.base_dataset.targets[self.indices[i]]
            self.classwise_indices[y].append(i)
        self.num_classes = max(self.classwise_indices.keys())+1

    def __getitem__(self, i):
        return self.base_dataset[self.indices[i]]

    def __len__(self):
        return len(self.indices)

    def get_class(self, i):
        return self.base_dataset.targets[self.indices[i]]


class ConcatWrapper(Dataset): # TODO: Naming
    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            l = len(e)
            r.append(l + s)
            s += l
        return r

    @staticmethod
    def numcls(sequence):
        s = 0
        for e in sequence:
            l = e.num_classes
            s += l
        return s

    @staticmethod
    def clsidx(sequence):
        r, s, n = defaultdict(list), 0, 0
        for e in sequence:
            l = e.classwise_indices
            for c in range(s, s + e.num_classes):
                t = np.asarray(l[c-s]) + n
                r[c] = t.tolist()
            s += e.num_classes
            n += len(e)
        return r

    def __init__(self, datasets):
        super(ConcatWrapper, self).__init__()
        assert len(datasets) > 0, 'datasets should not be an empty iterable'
        self.datasets = list(datasets)
        # for d in self.datasets:
        #     assert not isinstance(d, IterableDataset), "ConcatDataset does not support IterableDataset"
        self.cumulative_sizes = self.cumsum(self.datasets)

        self.num_classes = self.numcls(self.datasets)
        self.classwise_indices = self.clsidx(self.datasets)

    def __len__(self):
        return self.cumulative_sizes[-1]

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx]

    def get_class(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        true_class = self.datasets[dataset_idx].base_dataset.targets[self.datasets[dataset_idx].indices[sample_idx]]
        return self.datasets[dataset_idx].base_dataset.target_transform(true_class)

    @property
    def cummulative_sizes(self):
        warnings.warn("cummulative_sizes attribute is renamed to "
                      "cumulative_sizes", DeprecationWarning, stacklevel=2)
        return self.cumulative_sizes



def load_dataset(name, root, sample='default', **kwargs):
    # Dataset
    if name in ['imagenet','tinyimagenet', 'CUB200', 'STANFORD120', 'MIT67']:
        # TODO
        if name == 'tinyimagenet':
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(32),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
            transform_test = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

            train_val_dataset_dir = os.path.join(root, "train")
            test_dataset_dir = os.path.join(root, "val")

            trainset = DatasetWrapper(datasets.ImageFolder(root=train_val_dataset_dir, transform=transform_train))
            valset   = DatasetWrapper(datasets.ImageFolder(root=test_dataset_dir, transform=transform_test))

        elif name == 'imagenet':
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
            transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
            train_val_dataset_dir = os.path.join(root, "train")
            test_dataset_dir = os.path.join(root, "val")

            trainset = DatasetWrapper(datasets.ImageFolder(root=train_val_dataset_dir, transform=transform_train))
            valset   = DatasetWrapper(datasets.ImageFolder(root=test_dataset_dir, transform=transform_test))

        else:
            transform_train = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(), 
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
            transform_test = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])

            train_val_dataset_dir = os.path.join(root, name, "train")
            test_dataset_dir = os.path.join(root, name, "test")

            trainset = DatasetWrapper(datasets.ImageFolder(root=train_val_dataset_dir, transform=transform_train))
            valset   = DatasetWrapper(datasets.ImageFolder(root=test_dataset_dir, transform=transform_test))

    elif name.startswith('cifar'):
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        if name == 'cifar10':
            CIFAR = datasets.CIFAR10
        else:
            CIFAR = datasets.CIFAR100

        trainset = DatasetWrapper(CIFAR(root, train=True,  download=True, transform=transform_train))
        valset   = DatasetWrapper(CIFAR(root, train=False, download=True, transform=transform_test))
    
####################
    elif name == 'hospital_data':
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        transform_test = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        train_val_dataset_dir = os.path.join(root, "train")
        test_dataset_dir = os.path.join(root, "val")
        trainset = DatasetWrapper(datasets.ImageFolder(root=train_val_dataset_dir, transform=transform_train))
        valset   = DatasetWrapper(datasets.ImageFolder(root=test_dataset_dir, transform=transform_test))
    else:
        raise Exception('Unknown dataset: {}'.format(name))

    # Sampler
    if sample == 'default':
        get_train_sampler = lambda d: BatchSampler(RandomSampler(d), kwargs['batch_size'], False)
        get_test_sampler  = lambda d: BatchSampler(SequentialSampler(d), kwargs['batch_size'], False)

    elif sample == 'pair':
        get_train_sampler = lambda d: PairBatchSampler(d, kwargs['batch_size'])
        get_test_sampler  = lambda d: BatchSampler(SequentialSampler(d), kwargs['batch_size'], False)

    else:
        raise Exception('Unknown sampling: {}'.format(sampling))

    trainloader = DataLoader(trainset, batch_sampler=get_train_sampler(trainset), num_workers=4)
    valloader   = DataLoader(valset,   batch_sampler=get_test_sampler(valset), num_workers=4)

    return trainloader, valloader

