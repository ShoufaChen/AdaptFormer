import os
from util.crop import RandomResizedCrop
from timm.data.constants import IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD, IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torchvision.transforms as transforms
import torchvision.datasets as datasets


def build_image_dataset(args):
    # linear probe: weak augmentation
    if os.path.basename(args.finetune).startswith('jx_'):
        _mean = IMAGENET_INCEPTION_MEAN
        _std = IMAGENET_INCEPTION_STD
    elif os.path.basename(args.finetune).startswith('mae_pretrain_vit'):
        _mean = IMAGENET_DEFAULT_MEAN
        _std = IMAGENET_DEFAULT_STD
    elif os.path.basename(args.finetune).startswith('swin_base_patch4'):
        _mean = IMAGENET_DEFAULT_MEAN
        _std = IMAGENET_DEFAULT_STD
    else:
        raise ValueError(os.path.basename(args.finetune))
    transform_train = transforms.Compose([
        RandomResizedCrop(224, interpolation=3),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=_mean, std=_std)])
    transform_val = transforms.Compose([
        transforms.Resize(256, interpolation=3),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=_mean, std=_std)])

    if args.dataset == 'imagenet':
        dataset_train = datasets.ImageFolder(os.path.join(args.data_path, 'train'), transform=transform_train)
        dataset_val = datasets.ImageFolder(os.path.join(args.data_path, 'val'), transform=transform_val)
        nb_classes = 1000

    elif args.dataset == 'cifar100':
        dataset_train = datasets.CIFAR100(os.path.join(args.data_path, 'cifar100'), transform=transform_train, train=True)
        dataset_val = datasets.CIFAR100(os.path.join(args.data_path, 'cifar100'), transform=transform_val, train=False)
        nb_classes = 100
    elif args.dataset == 'flowers102':
        from flowers102 import Flowers102
        dataset_train = Flowers102(os.path.join(args.data_path, 'flowers102'), split='train', transform=transform_train)
        dataset_val = Flowers102(os.path.join(args.data_path, 'flowers102'), split='test', transform=transform_val)
        nb_classes = 102
    elif args.dataset == 'svhn':
        from torchvision.datasets import SVHN
        dataset_train = SVHN(os.path.join(args.data_path, 'svhn'), split='train', transform=transform_train)
        dataset_val = SVHN(os.path.join(args.data_path, 'svhn'), split='test', transform=transform_val)
        nb_classes = 10
    elif args.dataset == 'food101':
        from .food101 import Food101
        dataset_train = Food101(os.path.join(args.data_path, 'food101'), split='train', transform=transform_train)
        dataset_val = Food101(os.path.join(args.data_path, 'food101'), split='test', transform=transform_val)
        nb_classes = 101
    else:
        raise ValueError(args.dataset)

    return dataset_train, dataset_val, nb_classes
