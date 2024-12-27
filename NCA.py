import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

class NeuralCA(nn.Module):
    def __init__(self, channels, hidden_dim=128):
        super(NeuralCA, self).__init__()
        self.channels = channels
        self.hidden_dim = hidden_dim
        
        self.perceive = nn.Conv2d(channels, hidden_dim, kernel_size=3, padding=1, bias=False)
        self.update = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(hidden_dim, channels, kernel_size=1, bias=False)
        )

    def forward(self, x):
        dx = self.perceive(x)
        dx = self.update(dx)
        return x + dx

def seed_grid(grid_size, seed_size, channels):
    grid = np.zeros((channels, grid_size, grid_size), dtype=np.float32)
    center = grid_size // 2
    grid[:, center-seed_size:center+seed_size, center-seed_size:center+seed_size] = 1.0
    return grid

def train(model, target, steps=1000, lr=1e-3, device='cuda'):
    target = torch.tensor(target, dtype=torch.float32).unsqueeze(0).to(device)
    grid_size = target.shape[-1]

    optimizer = optim.Adam(model.parameters(), lr=lr)
    grid = seed_grid(grid_size, seed_size=3, channels=model.channels)
    grid = torch.tensor(grid, dtype=torch.float32).unsqueeze(0).to(device)

    for step in range(steps):
        optimizer.zero_grad()
        
        for _ in range(np.random.randint(1, 4)):  # Apply 1-3 random steps
            grid = model(grid)

        loss = torch.mean((grid - target)**2)
        loss.backward()
        optimizer.step()

        if step % 100 == 0 or step == steps - 1:
            print(f"Step {step}/{steps}, Loss: {loss.item():.6f}")

    return grid

if __name__ == "__main__":
    grid_size = 64
    channels = 16
    hidden_dim = 128
    target = seed_grid(grid_size, seed_size=10, channels=channels)
    model = NeuralCA(channels, hidden_dim).to('cuda')

    final_grid = train(model, target, steps=1000, lr=1e-3, device='cuda')

    plt.imshow(final_grid.squeeze().detach().cpu().numpy().transpose(1, 2, 0))
    plt.title("Final Grid State")
    plt.axis('off')
    plt.show()
