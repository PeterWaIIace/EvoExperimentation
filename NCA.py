import numpy as np
import matplotlib.pyplot as plt
from cmaes import CMA_ES 
import torch
import torch.nn as nn
import torch.optim as optim

class NeuralCA(nn.Module):
    def __init__(self, channels, weights, hidden_dim=128):
        super(NeuralCA, self).__init__()
        self.channels = channels
        self.hidden_dim = hidden_dim

        self.perceive = nn.Conv2d(channels, hidden_dim, kernel_size=3, padding=1, bias=False)
        self.update = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(hidden_dim, channels, kernel_size=1, bias=False)
        )

        self.perceive.weight.copy_(weights[0])
        self.update[1].weight.copy_(weights[1])

    def forward(self, x):
        dx = self.perceive(x)
        dx = self.update(dx)
        return x + dx

def seed_grid(grid_size, seed_size, channels):
    grid = np.zeros((channels, grid_size, grid_size), dtype=np.float32)
    center = grid_size // 2
    grid[:, center-seed_size:center+seed_size, center-seed_size:center+seed_size] = 1.0
    return grid


if __name__ == "__main__":
    CMA_ES

    # grid_size = 64
    # channels = 16
    # hidden_dim = 128
    # target = seed_grid(grid_size, seed_size=10, channels=channels)

    # model = NeuralCA(channels, hidden_dim).to('cuda')

    # plt.imshow(final_grid.squeeze().detach().cpu().numpy().transpose(1, 2, 0))
    # plt.title("Final Grid State")
    # plt.axis('off')
    # plt.show()
