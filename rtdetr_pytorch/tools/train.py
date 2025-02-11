import os 
import sys 
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import argparse

import src.misc.dist as dist 
from src.core import YAMLConfig 
from src.solver import TASKS
from nn_backbone_utils import ASPP
from src_zoo_rtdetr_hybrid_encoder import HybridEncoder

def main(args) -> None:
    dist.init_distributed()
    if args.seed is not None:
        dist.set_seed(args.seed)

    assert not all([args.tuning, args.resume]), 'Only support from_scratch or resume or tuning at one time'

    cfg = YAMLConfig(
        args.config,
        resume=args.resume, 
        use_amp=args.amp,
        tuning=args.tuning
    )

    solver = TASKS[cfg.yaml_cfg['task']](cfg)
    
    # Load ASPP + CBAM modifications into the model
    solver.model.backbone = ASPP(in_channels=1024, out_channels=256)
    solver.model.encoder = HybridEncoder()
    
    if args.test_only:
        solver.val()
    else:
        solver.fit()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', '-c', type=str)
    parser.add_argument('--resume', '-r', type=str)
    parser.add_argument('--tuning', '-t', type=str)
    parser.add_argument('--test-only', action='store_true', default=False)
    parser.add_argument('--amp', action='store_true', default=False)
    parser.add_argument('--seed', type=int, help='seed')
    args = parser.parse_args()

    main(args)
