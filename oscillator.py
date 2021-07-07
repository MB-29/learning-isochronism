import numpy as np
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchdyn.models import NeuralDE, NeuralSDE


class Oscillator(pl.LightningModule):
    def __init__(self, dynamics: nn.Module, time_interval):
        super().__init__()
        self.dynamics = dynamics
        self.model = dynamics.control
        self.neural_DE = NeuralDE(
            dynamics,
            sensitivity='adjoint',
            s_span=time_interval,
            solver='dopri5'
            )
        self.time_interval = time_interval
        
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):

        batch_final_state = self.neural_DE(batch) 

        loss = nn.MSELoss()(batch, batch_final_state)
        self.log('train_loss', loss)
        return loss
 
    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=0.01)

