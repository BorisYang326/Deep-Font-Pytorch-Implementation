from dataclasses import dataclass
import torch


INPUT_SIZE = 105
VFR_FONTS_NUM = 2383
SYN_DATA_COUNT_PER_FONT = 1000
SQUEEZE_RATIO = 2.5
@dataclass
class TrainConfig:
    num_classes: int = 2383
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 4096
    epochs: int = 20
    num_workers: int = 4
    prefetch_factor: int = 2
    lr: float = 1e-4
    weight_decay: float = 1e-5
    gradient_log_iter: int = 100
    
