from torchvision import datasets, transforms
from torch.utils.data import Dataset
import torch
import numpy as np
from utils.augmentors import Cutout, RainbowOut, ChannelSwap



class CustomDataset(Dataset):
    def __init__(self,
                 which_set='Cifar-10',
                 root=None,
                 train=True,
                 download=True,
                 return_idxs=False,
                 num_classes=10,
                 aug=('random_order', 'random_h_flip', 'random_crop', 'random_rot_10', 'random_scale_0.9_1.1', 'random_shear_5', 'cutout'),
                 cut_n_holes=1,
                 cut_length=16,
                 ):
        assert which_set in ['Cifar-10', 'MNIST', 'Fashion-MNIST'], \
            "Must be Cifar, MNIST, or Fashion MNIST, {} not allowed".format(which_set)

        image_length = 32 if which_set == 'Cifar-10' else 28
        normalizer = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        if which_set == 'MNIST':
            normalizer = transforms.Normalize((0.1307,), (0.3081,))
        elif which_set == 'Fashion-MNIST':
            normalizer = transforms.Normalize((0.1307,), (0.3081,))


        transforms_list = []
        for augment in aug:
            ############### First do the things that don't change where the image is in the box
            if augment == 'random_h_flip':
                transforms_list.append(transforms.RandomHorizontalFlip())
            if augment == 'random_v_flip':
                transforms_list.append(transforms.RandomVerticalFlip())
            ############### Then mess with brightness etc.
            if augment == 'color_jitter':
                transforms_list.append(transforms.ColorJitter(brightness=0.3, contrast=0.3, hue=0.0))
            ############### Now do some sheering/cropping/rotation that changes where the images is
            if augment == 'affine':
                rot_degrees = 0
                scale_low = None
                scale_high = None
                shear_degrees = None
                for augment_inner in aug:
                    if 'random_rot' in augment_inner:
                        rot_degrees = int(augment_inner.split('_')[-1])
                    if 'random_scale' in augment_inner:
                        scale_low = float(augment_inner.split('_')[-2])
                        scale_high = float(augment_inner.split('_')[-1])
                    if 'random_shear' in augment_inner:
                        shear_degrees = int(augment_inner.split('_')[-1])


                transforms_list.append(transforms.RandomAffine(degrees=rot_degrees,
                                                               scale=None if (scale_low == None) or (scale_high == None) else (scale_low, scale_high),
                                                               shear=shear_degrees))
            if augment == 'random_crop':
                transforms_list.append(transforms.RandomCrop(size=[image_length, image_length], padding=4))


        transform = transforms.Compose(transforms_list) if 'random_order' not in aug else transforms.Compose([transforms.RandomOrder(transforms=transforms_list)])

        transform.transforms.append(transforms.ToTensor())
        transform.transforms.append(normalizer)

        for augment in aug:
            ############### Finally do things that are related to regularisation
            if augment == 'cutout':
                transform.transforms.append(Cutout(n_holes=cut_n_holes, length=cut_length))
            if augment == 'rainbowout':
                transform.transforms.append(RainbowOut())
            if augment == 'channelswap':
                transform.transforms.append(ChannelSwap())

        if which_set == 'MNIST':
            self.dataset = datasets.MNIST(root='../data/MNIST' if root is None else root,
                                          train=train,
                                          download=download,
                                          transform=transform)

        elif which_set == 'Fashion-MNIST':
            self.dataset = datasets.FashionMNIST(root='../data/Fashion-MNIST' if root is None else root,
                                                 train=train,
                                                 download=download,
                                                 transform=transform)

        else:
            self.dataset = datasets.CIFAR10(root='../data/Cifar-10' if root is None else root,
                                            train=train, download=download,
                                            transform=transform)

        self.return_idxs = return_idxs
        self.num_classes = num_classes
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def __getitem__(self, index):
        data, target = self.dataset[index]

        if not self.return_idxs:
            return data, target
        else:
            return data, target, index

    def __len__(self):
        return len(self.dataset)


def load_dataset(args):
    print('==> Preparing data..')
    assert args.dataset in ['Cifar-10', 'Fashion-MNIST', 'MNIST'], "dataset {} not supported".format(args.dataset)
    in_shape = (28, 28, 1) if args.dataset != 'Cifar-10' else (32, 32, 3)

    train_dataset = CustomDataset(which_set=args.dataset,
                                  root='../data/{}'.format(args.dataset),
                                  train=True,
                                  download=True,
                                  aug=args.data_aug)
    trainloader = torch.utils.data.DataLoader(train_dataset,
                                              batch_size=args.batch_size, shuffle=True)

    test_dataset = CustomDataset(which_set=args.dataset,
                                  root='../data/{}'.format(args.dataset),
                                  train=False,
                                  download=True,
                                  aug=[])
    testloader = torch.utils.data.DataLoader(test_dataset,
                                             batch_size=args.test_batch_size, shuffle=False)

    return trainloader, testloader, in_shape

