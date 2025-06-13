import torchvision.models as models
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import CIFAR10, CIFAR100
from torch.utils.data import random_split
import torch

def loadData(used_batch_size=200):
    
   # Transform to preprocess the dataset (converting it to a tensor and normalizing it)
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.247, 0.243, 0.261])        # Normalization statistics for CIFAR-10 dataset
    ])

    # Load CIFAR10 dataset
    training_data = CIFAR10(root="dataset/", train=True, download=True, transform=transform)
    test_data = CIFAR10(root="dataset/", train=False, download=True, transform=transform)

    training_loader = torch.utils.data.DataLoader(batch_size=used_batch_size, dataset=training_data, shuffle=True)
    test_loader = torch.utils.data.DataLoader(batch_size=used_batch_size, dataset=test_data, shuffle=True)
    
    return training_loader, test_loader