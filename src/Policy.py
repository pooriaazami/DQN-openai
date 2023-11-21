import torch
import torch.nn as nn
import torch.nn.functional as F

class PolicyNetwork(nn.Module):
    def __init__(self, num_valid_actions=4):
        super().__init__()

        self.network = nn.Sequential(
            nn.Conv2d(in_channels=4, out_channels=16, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Linear(in_features=10, out_features=256),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(in_features=256, out_features=num_valid_actions),
        )
        