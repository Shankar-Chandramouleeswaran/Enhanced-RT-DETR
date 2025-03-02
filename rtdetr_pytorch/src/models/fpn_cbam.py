import torch
import torch.nn as nn
import torch.nn.functional as F

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction, bias=False),
            nn.ReLU(),
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
        # x: [B, C, H, W]
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

class FPN_CBAM(nn.Module):
    def __init__(self, in_channels_list, out_channels):
        """
        Args:
            in_channels_list (list[int]): number of channels from each backbone feature (e.g. [128, 256, 512])
            out_channels (int): target number of channels (should match the encoder hidden dimension)
        """
        super(FPN_CBAM, self).__init__()
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        self.cbam_blocks = nn.ModuleList()

        for in_channels in in_channels_list:
            # Lateral projection with a 1x1 convolution.
            self.lateral_convs.append(
                nn.Conv2d(in_channels, out_channels, kernel_size=1)
            )
            # FPN smoothing using a 3x3 convolution.
            self.fpn_convs.append(
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
            )
            # CBAM block to refine the feature.
            self.cbam_blocks.append(
                CBAM(out_channels)
            )
    
    def forward(self, inputs):
        """
        Args:
            inputs (list[Tensor]): list of backbone feature maps (e.g., [C3, C4, C5])
        Returns:
            List[Tensor]: refined feature maps
        """
        # Apply lateral convs.
        lateral_feats = [l_conv(x) for x, l_conv in zip(inputs, self.lateral_convs)]
        
        # Build the top–down pathway.
        out_feats = [None] * len(lateral_feats)
        x = lateral_feats[-1]
        out_feats[-1] = self.cbam_blocks[-1](self.fpn_convs[-1](x))
        for i in range(len(lateral_feats) - 2, -1, -1):
            x = F.interpolate(x, size=lateral_feats[i].shape[-2:], mode='nearest')
            x = lateral_feats[i] + x
            out_feats[i] = self.cbam_blocks[i](self.fpn_convs[i](x))
        return out_feats
