import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    def __init__(self,h,w,fc_neuron,num_classes=None):
        super().__init__()
        self.h=h
        self.w=w
        self.fc_neuron=fc_neuron

        self.fc1 = nn.LazyLinear(self.fc_neuron)
        self.fc2 = nn.LazyLinear(self.fc_neuron * 2)
        self.fc3 = nn.LazyLinear(self.fc_neuron * 3)
        self.fc4 = nn.LazyLinear(self.h * self.w)

    def forward(self, x):
        y = F.leaky_relu(self.fc1(x), negative_slope=0.2)
        y = F.leaky_relu(self.fc2(y), negative_slope=0.2)
        y = F.leaky_relu(self.fc3(y), negative_slope=0.2)
        y = torch.tanh(self.fc4(y))
        return y

class Discriminator(nn.Module):
    def __init__(self,fc_neuron):
        super().__init__()
        self.fc_neuron=fc_neuron

        self.fc1 = nn.LazyLinear(3 * self.fc_neuron)
        self.fc2 = nn.LazyLinear(2 * self.fc_neuron)
        self.fc3 = nn.LazyLinear(self.fc_neuron)
        self.fc4 = nn.LazyLinear(1)
        
    def forward(self, x):
        y = F.dropout(F.leaky_relu(self.fc1(x), negative_slope=0.2), p=0.3)
        y = F.dropout(F.leaky_relu(self.fc2(y), negative_slope=0.2), p=0.3)
        y = F.dropout(F.leaky_relu(self.fc3(y), negative_slope=0.2), p=0.3)
        y = self.fc4(y)
        return y