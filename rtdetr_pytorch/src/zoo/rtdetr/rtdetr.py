"""by lyuwenyu
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import random 
import numpy as np 

from src.core import register
from ...models.aspp_cbam import ASPP_CBAM


__all__ = ['RTDETR', ]


@register
class RTDETR(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, backbone: nn.Module, encoder, decoder, multi_scale=None,
                 use_aspp_cbam=False, aspp_cbam_in_channels=None, aspp_cbam_out_channels=256):
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder
        self.multi_scale = multi_scale

        # Option to use ASPP+CBAM on each scale.
        self.use_aspp_cbam = use_aspp_cbam
        if self.use_aspp_cbam:
            if aspp_cbam_in_channels is None:
                raise ValueError("When using ASPP_CBAM, please provide 'aspp_cbam_in_channels' as a list (e.g., [512, 1024, 2048]).")
            # Create one ASPP_CBAM module per feature map scale.
            self.aspp_cbam_list = nn.ModuleList([
                ASPP_CBAM(in_ch, aspp_cbam_out_channels) for in_ch in aspp_cbam_in_channels
            ])
        
    def forward(self, x, targets=None):
        if self.multi_scale and self.training:
            sz = np.random.choice(self.multi_scale)
            x = F.interpolate(x, size=[sz, sz])
        
        # Get multi-scale features from the backbone.
        features = self.backbone(x)  # e.g., features = [F1, F2, F3]
        
        # If ASPP+CBAM is enabled, process each feature map.
        if self.use_aspp_cbam:
            features = [module(feat) for feat, module in zip(features, self.aspp_cbam_list)]
        
        encoded_feats = self.encoder(features)
        out = self.decoder(encoded_feats, targets)
        return out
    
    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self 
