import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self, noise_dim, h, w, fc_neuron):
        super().__init__()
        self.h = h
        self.w = w
        self.fc_neuron = fc_neuron

        # Assuming `noise_dim` is the size of the noise vector
        self.fc1 = nn.Linear(noise_dim, self.fc_neuron)
        self.bn1 = nn.BatchNorm1d(self.fc_neuron)
        self.fc2 = nn.Linear(self.fc_neuron, self.fc_neuron * 2)
        self.bn2 = nn.BatchNorm1d(self.fc_neuron * 2)
        self.fc3 = nn.Linear(self.fc_neuron * 2, self.fc_neuron * 3)
        self.bn3 = nn.BatchNorm1d(self.fc_neuron * 3)
        self.fc4 = nn.Linear(self.fc_neuron * 3, self.h * self.w)

    def forward(self, x):
        y = F.leaky_relu(self.bn1(self.fc1(x)), negative_slope=0.2)
        y = F.leaky_relu(self.bn2(self.fc2(y)), negative_slope=0.2)
        y = F.leaky_relu(self.bn3(self.fc3(y)), negative_slope=0.2)
        y = torch.tanh(self.fc4(y))
        return y.view(-1, 1, self.h, self.w)  # Reshape to match image dimensions

class Discriminator(nn.Module):
    def __init__(self, h, w, fc_neuron):
        super().__init__()
        self.fc_neuron = fc_neuron
        input_dim = h * w  # Input dimension based on image size

        self.fc1 = nn.Linear(input_dim, 3 * self.fc_neuron)
        self.d1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(3 * self.fc_neuron, 2 * self.fc_neuron)
        self.d2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(2 * self.fc_neuron, self.fc_neuron)
        self.d3 = nn.Dropout(0.3)
        self.fc4 = nn.Linear(self.fc_neuron, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        y = self.d1(F.leaky_relu(self.fc1(x), negative_slope=0.2))
        y = self.d2(F.leaky_relu(self.fc2(y), negative_slope=0.2))
        y = self.d3(F.leaky_relu(self.fc3(y), negative_slope=0.2))
        y = torch.sigmoid(self.fc4(y))
        return y
