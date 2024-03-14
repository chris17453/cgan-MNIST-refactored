import torch
from torchvision import transforms as T
from torchvision.datasets import EMNIST,MNIST
from PIL import Image
import numpy as np
from PIL import ImageFilter, Image
import cv2
from .data_transforms import CLAHETransform


class CustomTransform:
    def __init__(self, model_type):
        self.model_type = model_type
        if model_type == "MNIST":
            self.transform_data = T.Compose([
#                T.Grayscale(),  # Ensure image is grayscale
#                ConvertAndCombineWithBlur(threshold=0.5, radius=1),  # Custom transformation
#                T.Lambda(lambda img: conditional_average(img)),
#                T.Lambda(lambda img: anti_aliasing(img)),
                CLAHETransform(),
                T.ToTensor(),  # Convert back to tensor to continue with the pipeline
                T.Normalize(mean=0.5, std=.5),
                #T.Normalize((0.1307,), (0.3081,))
            ])
        elif model_type == "EMNIST":
            self.transform_data = T.Compose([
                lambda img: T.functional.rotate(img, -90),
                lambda img: T.functional.hflip(img),
                CLAHETransform(),
                T.ToTensor(),  # Convert back to tensor after processing
                T.Normalize(mean=0.5, std=.5)
            ])
    
    def load_datasets(self):
        if self.model_type == "MNIST":
            train_dataset = MNIST(root='data', train=True , transform=self.transform_data, download=True)
            valid_dataset = MNIST(root='data', train=False, transform=self.transform_data, download=True)
        elif self.model_type == "EMNIST":
            train_dataset = EMNIST(root='data', split='balanced', train=True, transform=self.transform_data, download=True)
            valid_dataset = EMNIST(root='data', split='balanced', train=False, transform=self.transform_data, download=True)
        return train_dataset, valid_dataset