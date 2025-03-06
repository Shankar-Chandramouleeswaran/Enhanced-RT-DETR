"""by lyuwenyu
"""

import torch 
import torch.nn as nn 
import torch.nn.functional as F 

import random 
import numpy as np 

from src.core import register
from ...models.spp_cbam import SPP_CBAM

__all__ = ['RTDETR', ]


@register
class RTDETR(nn.Module):
    __inject__ = ['backbone', 'encoder', 'decoder']
    def __init__(self, backbone: nn.Module, encoder, decoder, multi_scale=None,
                 use_spp_cbam=False, spp_cbam_in_channels=None, spp_cbam_out_channels=256):
        super().__init__()
        self.backbone = backbone
        self.encoder = encoder
        self.decoder = decoder
        self.multi_scale = multi_scale

        self.use_spp_cbam = use_spp_cbam
        if self.use_spp_cbam:
            if spp_cbam_in_channels is None:
                raise ValueError("When using SPP_CBAM, please provide 'spp_cbam_in_channels' as a list (e.g., [512, 1024, 2048]).")
            # Create one SPP_CBAM per scale.
            self.spp_cbam_list = nn.ModuleList([
                SPP_CBAM(in_ch, spp_cbam_out_channels) for in_ch in spp_cbam_in_channels
            ])

    def forward(self, x, targets=None):
        if self.multi_scale and self.training:
            sz = np.random.choice(self.multi_scale)
            x = F.interpolate(x, size=[sz, sz])
        
        features = self.backbone(x)  # e.g., features = [F1, F2, F3]

        if self.use_spp_cbam:
            # Process each feature map with the corresponding SPP_CBAM module.
            features = [module(feat) for feat, module in zip(features, self.spp_cbam_list)]

        encoded_feats = self.encoder(features)
        out = self.decoder(encoded_feats, targets)
        return out
    
    def deploy(self, ):
        self.eval()
        for m in self.modules():
            if hasattr(m, 'convert_to_deploy'):
                m.convert_to_deploy()
        return self 
