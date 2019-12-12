import torch
from torchvision import datasets, transforms
from torch.utils import data


class StandardImgData():
    def __init__(self, samples_per_class, batch_size, dataset,
                 unlab_samples_per_class=1000):

        self.root = 'data/%s/' % dataset
        self.dataset = dataset
        self.img_sz = 32
        self.samples_per_class = samples_per_class
        self.batch_size = batch_size
        self.unlab_samples_per_class = unlab_samples_per_class

        self.transform = transforms.Compose([
            transforms.Resize(self.img_sz),
            transforms.CenterCrop(self.img_sz),
            transforms.ToTensor(),
            transforms.Normalize(mean=[.5, ], std=[.5]) if dataset == 'mnist' \
                else transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]),
        ])

    def get_dataloaders(self, dataset):
        if dataset == 'mnist':
            full_dataset = datasets.MNIST(root=self.root, train=True, transform=self.transform, target_transform=None,
                                          download=True)
            test_dataset = datasets.MNIST(root=self.root, train=False, transform=self.transform, target_transform=None,
                                          download=True)
        elif dataset == 'cifar':
            full_dataset = datasets.CIFAR10(root=self.root, train=True, transform=self.transform, target_transform=None,
                                            download=True)
            test_dataset = datasets.CIFAR10(root=self.root, train=False, transform=self.transform,
                                            target_transform=None,
                                            download=True)

        train_size = int(0.8 * len(full_dataset))
        valid_size = len(full_dataset) - train_size
        train_dataset, valid_dataset = torch.utils.data.random_split(full_dataset, [train_size, valid_size])

        train_unl_dataset = self.__get_samples_per_class(train_dataset, self.unlab_samples_per_class)
        train_lb_dataset = self.__get_samples_per_class(train_dataset, self.samples_per_class)

        train_unl_dataloader = data.DataLoader(train_unl_dataset, batch_size=self.batch_size, shuffle=True)
        train_lb_dataloader = data.DataLoader(train_lb_dataset, batch_size=self.batch_size, shuffle=True)
        train_lb_dataloader = self.__create_infinite_dataloader(train_lb_dataloader)
        valid_dataloader = data.DataLoader(valid_dataset, batch_size=self.batch_size, shuffle=False)
        test_dataloader = data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        return train_unl_dataloader, train_lb_dataloader, valid_dataloader, test_dataloader

    @staticmethod
    def __get_samples_per_class(dataset, num_samples):
        labels = torch.tensor([y for x, y in dataset])
        indices = torch.arange(len(labels))
        indices = torch.cat([indices[labels == x][:num_samples] for x in torch.unique(labels)])
        dataset = data.Subset(dataset, indices)
        return dataset

    @staticmethod
    def __create_infinite_dataloader(dataloader):
        data_iter = iter(dataloader)
        while True:
            try:
                yield next(data_iter)
            except StopIteration:
                data_iter = iter(dataloader)
