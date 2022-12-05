import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNetwork(nn.Module):
    def __init__(self, observations_size=(3, 256, 256), action_size=2, seed=0):
        super(PolicyNetwork, self).__init__()
        torch.manual_seed(seed=seed)

        self.observations_size = observations_size
        self.action_size = action_size
        channels, height, width = self.observations_size

        self.features =nn.Sequential(
                                        nn.Conv2d(in_channels=channels, out_channels=32, kernel_size=5, stride=1),
                                        nn.ReLU(),
                                        nn.MaxPool2d(3, 3),
                                        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1),
                                        nn.ReLU(),
                                        nn.MaxPool2d(3, 3),
                                        nn.ReLU(),
                                        nn.Flatten(),
                                        nn.Linear(64 * 27 * 27, 512),
                                        nn.ReLU(),
                                        nn.Linear(512, 128),
                                        nn.ReLU(),
                                    )


        self.actor = nn.Linear(128, self.action_size)

    def forward(self, state):
        x =  self.features(state)

        distribution = F.softmax(self.actor(x), dim=-1)

        return distribution