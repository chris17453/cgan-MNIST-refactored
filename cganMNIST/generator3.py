import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self,  h, w, fc_neuron,input_dim=10):
        super().__init__()
        self.h = h
        self.w = w
        self.fc_neuron = fc_neuron
        input_dim = h * w  # Assuming the input is a flattened image

        # Assuming input_dim is the dimensionality of the noise vector
        self.fc1 = nn.Linear(input_dim, self.fc_neuron)
        self.fc2 = nn.Linear(self.fc_neuron, self.fc_neuron * 2)
        self.fc3 = nn.Linear(self.fc_neuron * 2, self.fc_neuron * 3)
        self.fc4 = nn.Linear(self.fc_neuron * 3, self.h * self.w)

    def forward(self, x):
        y = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        y = F.leaky_relu(self.fc2(y), negative_slope=0.2)
        y = F.leaky_relu(self.fc3(y), negative_slope=0.2)
        y = torch.tanh(self.fc4(y))
        return y
    
class Discriminator(nn.Module):
    def __init__(self, h, w, fc_neuron):
        super().__init__()
        input_dim = h * w  # Assuming the input is a flattened image
        self.fc_neuron = fc_neuron

        self.fc1 = nn.Linear(input_dim, 3 * self.fc_neuron)
        self.fc2 = nn.Linear(3 * self.fc_neuron, 2 * self.fc_neuron)
        self.fc3 = nn.Linear(2 * self.fc_neuron, self.fc_neuron)
        self.fc4 = nn.Linear(self.fc_neuron, 1)
        
    def forward(self, x):
        y = F.dropout(F.leaky_relu(self.fc1(x), negative_slope=0.2), p=0.3)
        y = F.dropout(F.leaky_relu(self.fc2(y), negative_slope=0.2), p=0.3)
        y = F.dropout(F.leaky_relu(self.fc3(y), negative_slope=0.2), p=0.3)
        y = torch.sigmoid(self.fc4(y))  # Assuming binary classification
        return y
