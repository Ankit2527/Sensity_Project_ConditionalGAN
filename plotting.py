from fmnist import fmnist
from torchvision.utils import make_grid
import numpy as np
import torch

def real_image_per_class_grid(data_dir, batch_size, num_workers):
    """
    Returns a grid of each class of FashionMNIST.
    Inputs:
        data_dir - Directory in which the FashionMNIST dataset should be downloaded. 
        batch_size - Batch size to use for the data loaders
        num_workers - Number of workers to use in the data loaders.
    """
    train_loader = fmnist(data_dir, batch_size, num_workers)
    dataset = train_loader.dataset
    all_labels = dataset.targets
    unique_labels = np.unique(all_labels)

    indices = []
    sampled = []
    for i, l in enumerate(unique_labels):
        if l not in sampled:
            indices.extend(np.random.choice(np.where(all_labels == l)[0], 1))
            sampled.append(l)

    images_list = []
    labels_list = []
    for i in indices:
        image = dataset.data[i]
        label = dataset.targets[i]
        images_list.append(image)
        labels_list.append(label)

    x1 = images_list[0]
    x2 = images_list[1]
    x3 = images_list[2]
    x4 = images_list[3]
    x5 = images_list[4]
    x6 = images_list[5]
    x7 = images_list[6]
    x8 = images_list[7]
    x9 = images_list[8]
    x10 = images_list[9]

    # Create a grid for real images
    x_stck = torch.stack((x1, x2, x3, x4, x5, x6, x7, x8, x9, x10))
    x_stck = x_stck.unsqueeze(1)
    images_stck = x_stck.cpu().float()
    grid_real = make_grid(images_stck, nrow = 10, normalize = True)
    
    return grid_real