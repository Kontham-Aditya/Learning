import torch
import torch.nn as nn
import torch.nn.functional as F

class CalibRefineNet(nn.Module):
    def __init__(self):
        super(CalibRefineNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)

        self.conv2 = nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2)
        self.bn2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(256)

        # Dynamically compute the output size for the FC layer
        dummy_input = torch.zeros(1, 3, 256, 832)
        x = self._forward_conv(dummy_input)
        flattened_size = x.view(1, -1).shape[1]

        self.fc = nn.Linear(flattened_size, 512)

    def _forward_conv(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return x

    def forward(self, x):
        x = self._forward_conv(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return x
