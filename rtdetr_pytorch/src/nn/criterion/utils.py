import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        ca = self.channel_attention(x)
        x = x * ca
        sa = self.spatial_attention(torch.cat([torch.mean(x, dim=1, keepdim=True), torch.max(x, dim=1, keepdim=True)[0]], dim=1))
        return x * sa

def focal_loss(pred, target, alpha=0.25, gamma=2.0):
    prob = torch.sigmoid(pred)
    pt = torch.where(target == 1, prob, 1 - prob)
    loss = -alpha * (1 - pt).pow(gamma) * torch.log(pt + 1e-8)
    return loss.mean()

def giou_loss(pred_boxes, target_boxes):
    inter_area = (torch.min(pred_boxes[:, 2], target_boxes[:, 2]) - torch.max(pred_boxes[:, 0], target_boxes[:, 0])) * \
                 (torch.min(pred_boxes[:, 3], target_boxes[:, 3]) - torch.max(pred_boxes[:, 1], target_boxes[:, 1]))
    union_area = (pred_boxes[:, 2] - pred_boxes[:, 0]) * (pred_boxes[:, 3] - pred_boxes[:, 1]) + \
                 (target_boxes[:, 2] - target_boxes[:, 0]) * (target_boxes[:, 3] - target_boxes[:, 1]) - inter_area
    iou = inter_area / (union_area + 1e-8)
    loss = 1 - iou
    return loss.mean()
