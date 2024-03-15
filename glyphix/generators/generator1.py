import torch
from torch import nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, h, w, base_neuron_count, num_classes):
        super().__init__()
        self.h = h
        self.w = w
        self.base_neuron_count = base_neuron_count
        self.num_classes = num_classes

        # Embedding for conditional input
        self.label_embedding = nn.Embedding(num_classes, num_classes)

        # Initial fully connected layers
        self.fc1 = nn.Sequential(
            nn.LazyLinear(self.base_neuron_count),
            nn.BatchNorm1d(self.base_neuron_count),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.fc2 = nn.Sequential(
            nn.LazyLinear(self.base_neuron_count * 2),
            nn.BatchNorm1d(self.base_neuron_count * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.fc3 = nn.Sequential(
            nn.LazyLinear(self.base_neuron_count * 3),
            nn.BatchNorm1d(self.base_neuron_count * 3),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # Final layer without batch normalization
        self.fc4 = nn.LazyLinear(self.h * self.w)

    def forward(self, noise, labels):
        # Embed labels and concatenate with noise
        labels = self.label_embedding(labels)
        x = torch.cat([noise, labels], -1)
        
        y = self.fc1(x)
        y = self.fc2(y)
        y = self.fc3(y)
        y = torch.tanh(self.fc4(y))
        
        # Reshape to image dimensions
        return y.view(-1, self.h, self.w)
    
class Discriminator(nn.Module):
    def __init__(self, h, w, base_neuron_count, num_classes):
        super().__init__()
        self.h = h
        self.w = w
        self.base_neuron_count = base_neuron_count
        self.num_classes = num_classes

        self.label_embedding = nn.Embedding(num_classes, h*w)

        self.model = nn.Sequential(
            nn.Linear(h*w + h*w, self.base_neuron_count * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.base_neuron_count * 2, self.base_neuron_count),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(self.base_neuron_count, 1),
            nn.Sigmoid(),
        )

    def forward(self, image, labels):
        # Flatten image
        image_flat = image.view(image.size(0), -1)
        # Embed labels to match image flattening, then concatenate
        labels = self.label_embedding(labels)
        labels = labels.view(labels.size(0), -1)
        x = torch.cat([image_flat, labels], -1)
        # Pass concatenated vector to model
        validity = self.model(x)
        return validity
