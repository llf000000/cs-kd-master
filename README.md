# Regularizing Class-wise Predictions via Self-knowledge Distillation (CS-KD)

PyTorch implementation of ["Regularizing Class-wise Predictions via Self-knowledge Distillation"](https://arxiv.org/abs/2003.13964) (CVPR 2020).

## Requirements

`torch==1.2.0`, `torchvision==0.4.0`

## MyUnderstanding
1、在将数据封装成一个batch的时候，将实际的batch分成了两部分，第一部分的batch从样本中随机选取batch_size个数量，第二部分的batch通过迭代上一部分的batch里的样本，根据每一个样本的类别又去数据集里随机选取与之标签一样的图片作为一对pair依次放入第一个batch里。所以实际的一个batch的大小相当于2*batch_size

2、在模型训练的时候，前面一半的数据正常通过网络模型得到logits输出，并得到由交叉熵损失函数计算得到的损失，后面一半的数据同样经过网络模型得到logits输出。将后一半batch的输出作为教师，前一半batch的输出作为学生，经过知识蒸馏技术得到第二类损失。最后第二类损失乘以指定系数加上交叉熵损失得到总损失。

3、如果需要使用自己的数据集，需要在train.py的配置信息部分进行修改，修改后的文件命名为train_hospital_data.py:

```py
parser.add_argument('--dataset', default='hospital_data', type=str, help='the name for dataset cifar100 | tinyimagenet | CUB200 | STANFORD120 | MIT67') # 数据集名称
parser.add_argument('--dataroot', default='../hospital_data/', type=str, help='data directory') # 数据目录
```

此外，需要在datasets.py文件里进行修改，修改后的文件命名为datasets_hospital_data.py：

```py
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
        valset = DatasetWrapper(datasets.ImageFolder(root=test_dataset_dir, transform=transform_test))
```

5、在dataset.py上运行后实验结果如下：

```
Best Accuracy : 49.5934944152832
```

## Run experiments

train cifar100 on resnet with class-wise regularization losses

`python3 train.py --sgpu 0 --lr 0.1 --epoch 200 --model CIFAR_ResNet18 --name test_cifar --decay 1e-4 --dataset cifar100 --dataroot ~/data/ -cls --lamda 1`

train fine-grained dataset on resnet with class-wise regularization losses

`python3 train.py --sgpu 0 --lr 0.1 --epoch 200 --model resnet18 --name test_cub200 --batch-size 32 --decay 1e-4 --dataset CUB200   --dataroot ~/data/ -cls --lamda 3`

## Citation
If you use this code for your research, please cite our papers.
```
@InProceedings{Yun_2020_CVPR,
author = {Yun, Sukmin and Park, Jongjin and Lee, Kimin and Shin, Jinwoo},
title = {Regularizing Class-Wise Predictions via Self-Knowledge Distillation},
booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
month = {June},
year = {2020}
}
```
