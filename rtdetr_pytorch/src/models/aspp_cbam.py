import torch
import torch.nn as nn
import torch.nn.functional as F

# We can reuse CBAM from our previous module
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels // reduction, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x: [B, C, H, W]
        avg_pool = F.adaptive_avg_pool2d(x, 1).view(x.size(0), x.size(1))
        max_pool = F.adaptive_max_pool2d(x, 1).view(x.size(0), x.size(1))
        avg_out = self.mlp(avg_pool)
        max_out = self.mlp(max_pool)
        scale = self.sigmoid(avg_out + max_out).unsqueeze(2).unsqueeze(3)
        return x * scale

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        padding = (kernel_size - 1) // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        avg_pool = torch.mean(x, dim=1, keepdim=True)
        max_pool, _ = torch.max(x, dim=1, keepdim=True)
        concat = torch.cat([avg_pool, max_pool], dim=1)
        scale = self.sigmoid(self.conv(concat))
        return x * scale

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
    
    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, dilations=[6, 12, 18], dropout=0.5):
        super(ASPP, self).__init__()
        # 1x1 conv branch
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # Atrous conv branches
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      padding=dilations[0], dilation=dilations[0], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      padding=dilations[1], dilation=dilations[1], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3,
                      padding=dilations[2], dilation=dilations[2], bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # Global pooling branch
        self.branch5 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        # Fuse branches
        self.conv1 = nn.Conv2d(out_channels * 5, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        size = x.shape[2:]
        feat1 = self.branch1(x)
        feat2 = self.branch2(x)
        feat3 = self.branch3(x)
        feat4 = self.branch4(x)
        feat5 = self.branch5(x)
        feat5 = F.interpolate(feat5, size=size, mode='bilinear', align_corners=False)
        x = torch.cat([feat1, feat2, feat3, feat4, feat5], dim=1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x

class ASPP_CBAM(nn.Module):
    def __init__(self, in_channels, out_channels, dilations=[6, 12, 18], dropout=0.5):
        super(ASPP_CBAM, self).__init__()
        self.aspp = ASPP(in_channels, out_channels, dilations=dilations, dropout=dropout)
        self.cbam = CBAM(out_channels)
    def forward(self, x):
        x = self.aspp(x)
        x = self.cbam(x)
        return x
