import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import pytorch_lightning as pl

from dynamics import *
from oscillator import Oscillator


n_training = 200
batch_size = 50
n_epochs = 5
Nt = 100
T = 2*np.pi
time_interval = torch.linspace(0, T, Nt)


dataset = 2*torch.randn((n_training, 2))
train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
batch = next(iter(train_dataloader))

control = nn.Sequential(
    nn.Linear(2, 16),
    nn.Tanh(),
    nn.Linear(16, 1)
)

dynamics = Dynamics(control)


# TRAIN

oscillator = Oscillator(dynamics, time_interval)
trainer = pl.Trainer(
    min_epochs=2,
    max_epochs=n_epochs
)
trainer.fit(oscillator, train_dataloader)

# PLOT
n_pts = 50
x = torch.linspace(-1, 1, n_pts)
y = torch.linspace(-1, 1, n_pts)
X, Y = torch.meshgrid(x, y)
z = torch.cat([X.reshape(-1, 1), Y.reshape(-1, 1)], 1)
f = dynamics(z).detach()

fx, fy = f[:, 0], f[:, 1]
fx, fy = fx.reshape(n_pts, n_pts), fy.reshape(n_pts, n_pts)
magnitude = np.sqrt(fx.numpy().T**2 + fy.numpy().T**2)
linewidth = magnitude / magnitude.max()
plt.streamplot(X.numpy().T, Y.numpy().T, fx.numpy().T,
               fy.numpy().T, color='black', linewidth=linewidth)
plt.xticks([-1, 0, 1])
plt.yticks([-1, 0, 1])
plt.xlabel(r'$x$')
plt.ylabel(r'$\dot{x}$')

plt.show()
