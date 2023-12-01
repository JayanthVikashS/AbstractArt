import torch
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import SubsetRandomSampler
from torch.utils.data import sampler


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


class ChunkSampler(sampler.Sampler):
    def __init__(self, num_samples, start=0):
        self.num_samples = num_samples
        self.start = start

    def __iter__(self):
        return iter(range(self.start, self.start + self.num_samples))

    def __len__(self):
        return self.num_samples


def get_dataloaders_abstract(batch_size, num_workers=0,
                            validation_fraction=None,
                            train_transforms=None, test_transforms=None):

    if train_transforms is None:
        train_transforms = transforms.ToTensor()

    if test_transforms is None:
        test_transforms = transforms.ToTensor()

    train_dataset = datasets.ImageFolder(root='Abstract/',
                                     transform=train_transforms,
    )

    valid_dataset = datasets.ImageFolder(root='AbstractValid/',
                                     transform=train_transforms,
    )

    test_dataset = datasets.ImageFolder(root='AbstractTest/',
                                     transform=train_transforms,
    )

    train_loader = DataLoader(dataset=train_dataset,
    		                  batch_size=batch_size,
    		                  num_workers=num_workers,
    		                  shuffle=True)
		                  
    valid_loader = DataLoader(dataset=valid_dataset,
		                  batch_size=batch_size,
		                  num_workers=num_workers,
		                  shuffle=True)	                  

    test_loader = DataLoader(dataset=test_dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             shuffle=False)

    return train_loader, valid_loader, test_loader
