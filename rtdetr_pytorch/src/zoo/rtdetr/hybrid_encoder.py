import torch
import torch.nn as nn
import torch.nn.functional as F
from nn_backbone_utils import ASPP
from nn_criterion_utils import CBAM

class HybridEncoder(nn.Module):
    def __init__(self, in_channels=[512, 1024, 2048], hidden_dim=256):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_dim = hidden_dim

        # Adding ASPP before multi-scale feature fusion
        self.aspp = ASPP(in_channels=1024, out_channels=256)  

        # Multi-scale feature fusion layers
        self.conv1 = nn.Conv2d(512, hidden_dim, kernel_size=1)
        self.conv2 = nn.Conv2d(1024, hidden_dim, kernel_size=1)
        self.conv3 = nn.Conv2d(2048, hidden_dim, kernel_size=1)

        # Apply CBAM after feature fusion
        self.cbam = CBAM(hidden_dim)

    def forward(self, features):
        f1, f2, f3 = features

        # Apply ASPP
        f2 = self.aspp(f2)

        # Feature Fusion
        f1 = self.conv1(f1)
        f2 = self.conv2(f2)
        f3 = self.conv3(f3)

        fused_features = f1 + f2 + f3

        # Apply CBAM for refinement
        refined_features = self.cbam(fused_features)

        return refined_features
