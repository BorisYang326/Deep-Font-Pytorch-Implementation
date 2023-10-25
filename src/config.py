from dataclasses import dataclass
import torch


INPUT_SIZE = 105
VFR_FONTS_NUM = 2383
SYN_DATA_COUNT_PER_FONT = 1000
SQUEEZE_RATIO = 2.5
EVAL_SQUEEZE_RATIO_RANGE = (1.5, 3.5)
NUM_RANDOM_CROP = 5
# number of picking augmentation combinations
COMB_PICK_NUM = 3
# test details
SQUEEZE_RATIO_RANGE = (1.5, 3.5)
RATIO_SAMPLES = 3
PATCH_SAMPLES = 5


@dataclass
class TrainConfig:
    num_classes: int = 2383
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 4096
    epochs: int = 20
    num_workers: int = 16
    prefetch_factor: int = 2
    lr: float = 1e-4
    weight_decay: float = 1e-5
    gradient_log_iter: int = 100
