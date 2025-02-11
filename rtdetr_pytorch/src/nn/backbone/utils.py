

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from typing import Dict, List

class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels, rates=(1, 6, 12, 18)):
        super(ASPP, self).__init__()
        self.blocks = nn.ModuleList([
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=rate, dilation=rate, bias=False)
            for rate in rates
        ])
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)
    
    def forward(self, x):
        features = [block(x) for block in self.blocks]
        global_feat = self.global_pool(x)
        global_feat = F.interpolate(global_feat, size=x.shape[2:], mode='bilinear', align_corners=False)
        features.append(global_feat)
        return self.bn(torch.cat(features, dim=1))

class IntermediateLayerGetter(nn.ModuleDict):
    def __init__(self, model: nn.Module, return_layers: List[str]) -> None:
        if not set(return_layers).issubset([name for name, _ in model.named_children()]):
            raise ValueError("return_layers are not present in model. {}".format([name for name, _ in model.named_children()]))
        
        orig_return_layers = return_layers
        return_layers = {str(k): str(k) for k in return_layers}
        layers = OrderedDict()
        
        for name, module in model.named_children():
            layers[name] = module
            if name in return_layers:
                del return_layers[name]
            if not return_layers:
                break

        super().__init__(layers)
        self.return_layers = orig_return_layers
        self.aspp = ASPP(in_channels=1024, out_channels=256)  # Adjust in_channels as per backbone

    def forward(self, x):
        outputs = []
        for name, module in self.items():
            x = module(x)
            if name in self.return_layers:
                outputs.append(x)
        # Apply ASPP to extracted features
        x = self.aspp(x)
        outputs.append(x)
        return outputs
