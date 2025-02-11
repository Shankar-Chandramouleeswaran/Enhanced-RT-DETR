import torch 
import torch.nn as nn 
from datetime import datetime
from pathlib import Path 
from src.misc import dist
from src.core import BaseConfig

class BaseSolver(object):
    def __init__(self, cfg: BaseConfig) -> None:
        self.cfg = cfg 

    def setup(self):
        cfg = self.cfg
        device = cfg.device
        self.device = device
        self.last_epoch = cfg.last_epoch

        self.model = dist.warp_model(cfg.model.to(device), cfg.find_unused_parameters, cfg.sync_bn)
        self.criterion = cfg.criterion.to(device)
        self.postprocessor = cfg.postprocessor
        self.scaler = cfg.scaler
        self.ema = cfg.ema.to(device) if cfg.ema is not None else None 
        self.output_dir = Path(cfg.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def eval(self):
        self.setup()
        self.val_dataloader = dist.warp_loader(self.cfg.val_dataloader, shuffle=self.cfg.val_dataloader.shuffle)

    def val(self):
        self.eval()
        ap_small = self.compute_ap_small(self.model, self.val_dataloader)  # Compute AP for small objects
        print(f"AP_small: {ap_small:.4f}")

    def compute_ap_small(self, model, dataloader):
        small_objects = []
        for images, targets in dataloader:
            predictions = model(images)
            small_objects.extend([box for box in predictions if (box[2] - box[0]) * (box[3] - box[1]) < 32 * 32])
        precision = recall = len(small_objects) / max(len(targets), 1)
        ap_small = precision * recall  # Simplified for demonstration
        return ap_small
