from random import shuffle
import torch
from torch.utils.data import DataLoader,random_split,ConcatDataset
from torchvision import datasets,transforms
from torchvision.transforms.autoaugment import AutoAugmentPolicy
import sys
import numpy as np

transforms_dict = {

    'standard': transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ]),
    
    # data augmentations for supcon                                     
    'supcon' : transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ]),
    
    #auto augment
    'auto_augment':transforms.Compose([transforms.ToPILImage(),
                                   transforms.AutoAugment(AutoAugmentPolicy.CIFAR10),
                                   transforms.ToTensor()]),
    

    'none' : transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
    
    }

class TwoCropTransform:
    """Create two crops of the same image"""
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return [self.transform(x), self.transform(x)]

def get_dataset(dataset_name,batch_size,shuffling=True,val_set=True,train_augmentations='standard',two_crop=False):
    #given the dataset name and batch size, returns two dataloaders and the dataset properties in a dictionary
    #normalising data

    assert train_augmentations in transforms_dict.keys(), "Pick a valid transform for the train augmentations"

    transform_train = transforms_dict[train_augmentations]
    transform_test = transforms_dict['none']

    if two_crop:
        transform_train = TwoCropTransform(transform_train)

    data_props = dict()

    if dataset_name == 'CIFAR10':
        train_data = datasets.CIFAR10('../Data/CIFAR10', train=True, download=False, transform = transform_train)
        test_data = datasets.CIFAR10('../Data/CIFAR10', train=False, download=False, transform = transform_test)
        data_props['classes'] = 10
        data_props['n_values_train'] = len(train_data)
        data_props['n_values_test'] = len(test_data)
        data_props['n_channels'] = 3
        data_props['res'] = 32
        data_props['ptch_sz'] =  4
        data_props['meta'] = np.load('../Data/CIFAR10/cifar-10-batches-py/batches.meta',allow_pickle=True)

    elif dataset_name == 'CIFAR100':
        train_data = datasets.CIFAR100('../Data/CIFAR100', train=True, download=False, transform = transform_train)
        test_data = datasets.CIFAR100('../Data/CIFAR100', train=False, download=False, transform = transform_test)
        data_props['classes'] = 100
        data_props['n_values_train'] = len(train_data)
        data_props['n_values_test'] = len(test_data)
        data_props['n_channels'] = 3
        data_props['res'] = 32
        data_props['ptch_sz'] =  4
        data_props['meta'] = np.load('../Data/CIFAR100/cifar-100-python/meta',allow_pickle=True)

    elif dataset_name == 'SVHN':
        train_data = datasets.SVHN('../Data/SVHN', split='train', download=False, transform = transform_train)
        test_data = datasets.SVHN('../Data/SVHN', split='test', download=False, transform = transform_test)
        data_props['classes'] = 10
        data_props['n_values_train'] = len(train_data)
        data_props['n_values_test'] = len(test_data)
        data_props['n_channels'] = 3
        data_props['res'] = 32
        data_props['ptch_sz'] =  4

    elif dataset_name == 'DTD':
        #special transforms to resize
        transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Resize((32,32))
        ])

        transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.Resize((32,32))
        ])
        train_data = datasets.DTD('../Data/DTD', split='train', download=False, transform = transform_train)
        test_data_single = datasets.DTD('../Data/DTD', split='test', download=False, transform = transform_test)
        val_data = datasets.DTD('../Data/DTD', split='val', download=False, transform = transform_test)
        test_data = ConcatDataset([train_data,test_data_single,val_data])
        data_props['classes'] = 47
        data_props['n_values_train'] = len(train_data)
        data_props['n_values_test'] = len(test_data)
        data_props['n_channels'] = 3
        data_props['res'] = 640
        data_props['ptch_sz'] =  4

    elif dataset_name == 'tiny-imagenet':
        #special transforms to resize
        transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406] , std = [0.229, 0.224, 0.225]),
        transforms.Resize((32,32))
        ])

        transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406] , std = [0.229, 0.224, 0.225]),
        transforms.Resize((32,32))
        ])
        train_data = datasets.ImageFolder('../../../scratch/jd5u19/data/tiny-imagenet-200/train/', transform = transform_train)
        test_data = datasets.ImageFolder('../../../scratch/jd5u19/data/tiny-imagenet-200/val/', transform = transform_test)
        data_props['classes'] = 200
        data_props['n_values_train'] = len(train_data)
        data_props['n_values_test'] = len(test_data)
        data_props['n_channels'] = 3
        data_props['res'] = 64
        data_props['ptch_sz'] =  4

    else:
        sys.exit('Specify valid dataset')

    if val_set == True:
        #get val props 
        torch.manual_seed(43) #to make sure repeatable
        val_size = int(len(train_data)*0.1)
        train_size = len(train_data) - val_size    
        #update dictionary 
        data_props['n_values_train'] = train_size
        data_props['n_values_val'] = val_size
        
        #create dataloaders
        train_split, val_split = random_split(train_data, [train_size, val_size])
        train_loader = DataLoader(train_split,batch_size=batch_size,shuffle=shuffling)
        val_loader = DataLoader(val_split,batch_size=batch_size,shuffle=shuffling)
        test_loader =   DataLoader(test_data, batch_size=batch_size,shuffle=shuffling)

        return train_loader,val_loader,test_loader,data_props

    else:
        train_loader = DataLoader(train_data,batch_size=batch_size,shuffle=shuffling)
        test_loader = DataLoader(test_data, batch_size=batch_size,shuffle=shuffling)

        return train_loader,test_loader,data_props

def preprocessing_empty(x,y):
    return x,y,y

#default alpha maximises accuracy in paper
def mixup_data(x, y, alpha=20, beta=1.0):
    if alpha > 0 and beta > 0:
        lam = lam = np.random.beta(alpha, beta)
    else:
        lam = 1 #cross-entropy

    batch_size = x.size()[0]
    index = torch.randperm(batch_size).cuda()
    mixed_x = lam * x + (1 - lam) * x[index] 
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def preprocessing_regmixup(x,y):
    mixup_x, part_y_a, part_y_b, lam = mixup_data(x, y)
    #since y is present in the first part of both targets_a and targets_b
    #The loss computed only on that part of the tensor is equivalent to 
    #the cross-entropy loss.
    targets_a = torch.cat([y, part_y_a])
    targets_b = torch.cat([y,part_y_b])
    x = torch.cat([x, mixup_x], dim=0)

    return x,(targets_a,targets_b,lam),y
