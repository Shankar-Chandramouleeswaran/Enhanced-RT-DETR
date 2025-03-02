"""by lyuwenyu
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import random 
import numpy as np 

from src.core import register
from ...models.fpn_cbam import FPN_CBAM


__all__ = ['RTDETR', ]


@register
class RTDETR(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder', ]

    def __init__(self, backbone: nn.Module, encoder, decoder, multi_scale=None,
                 use_fpn_cbam=False, fpn_in_channels=None, fpn_out_channels=256):
        super().__init__()
        self.backbone = backbone
        self.decoder = decoder
        self.encoder = encoder
        self.multi_scale = multi_scale
        
        # New option: use FPN+CBAM to refine backbone features.
        self.use_fpn_cbam = use_fpn_cbam
        if self.use_fpn_cbam:
            if fpn_in_channels is None:
                raise ValueError("When using FPN_CBAM, 'fpn_in_channels' must be provided (e.g., [128, 256, 512]).")
            self.fpn_cbam = FPN_CBAM(fpn_in_channels, fpn_out_channels)
        
    def forward(self, x, targets=None):
        if self.multi_scale and self.training:
            sz = np.random.choice(self.multi_scale)
            x = F.interpolate(x, size=[sz, sz])

        # Get multi-scale features from the backbone.    
        features = self.backbone(x) # Expected to be a list (e.g., [C3, C4, C5])

        # If enabled, process features through FPN+CBAM.
        if self.use_fpn_cbam:
            features = self.fpn_cbam(features)
        
        encoded_feats = self.encoder(features)        
        out = self.decoder(encoded_feats, targets)

        return out
    
    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self 
