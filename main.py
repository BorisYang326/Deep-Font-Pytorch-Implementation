import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from src.preprocess import TRANSFORMS_SQUEEZE
import hydra
from omegaconf import DictConfig
from hydra.core.config_store import ConfigStore
from src.config import TrainConfig
from hydra.utils import instantiate


cs = ConfigStore.instance()
cs.store(group="training", name="base_training_default", node=TrainConfig)

# need to set hydra.job.chdir=True first for version 1.2
@hydra.main(config_path="config", config_name="main", version_base='1.2')
def main(cfg: DictConfig) -> None:
    ## Dataset ##
    vfr_dataset = instantiate(cfg.dataset)(transform=TRANSFORMS_SQUEEZE)
    if cfg.model == 'scae':
        train_loader = DataLoader(
            vfr_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=cfg.training.num_workers,
            pin_memory=True,
            prefetch_factor=cfg.training.prefetch_factor,
        )
        eval_loader = None
    else:
        train_size = int(0.9 * len(vfr_dataset))
        eval_size = len(vfr_dataset) - train_size
        supervised_train_dataset, supervised_eval_dataset = random_split(
            vfr_dataset, [train_size, eval_size], torch.Generator().manual_seed(42)
        )
        train_loader = DataLoader(
            supervised_train_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=cfg.training.num_workers,
            pin_memory=True,
            prefetch_factor=cfg.training.prefetch_factor,
        )
        eval_loader = DataLoader(
            supervised_eval_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=cfg.training.num_workers,
            pin_memory=True,
            prefetch_factor=cfg.training.prefetch_factor,
        )
    ## Model ##
    model = instantiate(cfg.model)(cfg.training.num_classes)
    optimizer = instantiate(cfg.optimizer)(params=model.parameters())
    ## Trainer ##
    criterion = nn.MSELoss() if cfg.model == 'scae' else nn.CrossEntropyLoss()
    trainer = instantiate(cfg.trainer)(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_loader=train_loader,
        eval_loader=eval_loader,
    )
    ## Training ##
    trainer._train(cfg.training.epochs)
    trainer._writer.close()


if __name__ == '__main__':
    main()
