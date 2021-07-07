import torch
import torch.nn as nn



class Dynamics(nn.Module):

    def __init__(self, control):
        super(Dynamics, self).__init__()

        self.control = control

    def forward(self, x):
        force = self.control(x)
        return torch.cat((x[:, 1].unsqueeze(1), force), dim=1)
