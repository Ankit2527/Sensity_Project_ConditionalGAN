import torch
import torchvision
from torchvision import transforms
import torch.utils.data as data

def fmnist(data_dir, batch_size, num_workers):
    """
    Returns data loader for FashionMNIST dataset.
    Inputs:
        data_dir - Directory in which the FashionMNIST dataset should be downloaded. 
        batch_size - Batch size to use for the data loaders
        num_workers - Number of workers to use in the data loaders.
    """
    data_transforms = transforms.Compose([transforms.ToTensor()])
    data_loader = torchvision.datasets.FashionMNIST(root = data_dir, train = True, download = True, transform = data_transforms) 
    train_loader = data.DataLoader(data_loader, batch_size = batch_size, shuffle = True, num_workers = num_workers, pin_memory = True)
    return train_loader