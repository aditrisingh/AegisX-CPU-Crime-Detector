# ucf2/mc3.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class MC3(nn.Module):
    def __init__(self):
        super(MC3, self).__init__()
        self.conv1 = nn.Conv3d(3, 64, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.conv2 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)
        self.conv3 = nn.Conv3d(128, 256, kernel_size=(3, 3, 3), stride=1, padding=1)
        self.pool3 = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.feature_dim = 256  # Output after pooling

    def forward(self, x):
        x = F.relu(self.pool1(self.conv1(x)))
        x = F.relu(self.pool2(self.conv2(x)))
        x = F.relu(self.pool3(self.conv3(x)))
        x = x.view(x.size(0), -1)  # Flatten to [B, 256]
        return x  # Return features, no classification head

def get_mc3_feature_extractor():
    return MC3()
