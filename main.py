import torch.nn as nn
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig
from hydra.core.config_store import ConfigStore
from src.config import TrainConfig
from hydra.utils import instantiate
import logging
from src.preprocess import TRANSFORMS_STORE,collate_fn_PIL

logger = logging.getLogger(__name__)

cs = ConfigStore.instance()
cs.store(group="training", name="base_training_default", node=TrainConfig)


# need to set hydra.job.chdir=True first for version 1.2
@hydra.main(config_path="config", config_name="main", version_base='1.2')
def main(cfg: DictConfig) -> None:
    if cfg.model == 'scae':
        cfg.dataset.hdf5_file_path = cfg.dataset.train_hdf5_file_path
        unsupervised_train_dataset = instantiate(cfg.dataset)(TRANSFORMS_STORE[cfg.dataset.transforms])
        train_loader = DataLoader(
            unsupervised_train_dataset,
            batch_size=cfg.training.batch_size,
            shuffle=True,
            num_workers=cfg.training.num_workers,
            pin_memory=True,
            prefetch_factor=cfg.training.prefetch_factor,
            # collate_fn=custom_collate_fn,
        )
        eval_loader = None
    else:
        cfg.dataset.hdf5_file_path = cfg.train_hdf5_file_path
        supervised_train_dataset = instantiate(cfg.dataset)
        supervised_train_dataset.transform = TRANSFORMS_STORE[cfg.transforms]
        cfg.dataset.hdf5_file_path = cfg.eval_hdf5_file_path
        supervised_eval_dataset = instantiate(cfg.dataset)
        supervised_eval_dataset.transform = TRANSFORMS_STORE['eval']
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
    optim_groups = model._optim_groups(cfg.training.lr)
    optimizer = instantiate(cfg.optimizer)(params=optim_groups)
    ## Trainer ##
    criterion = nn.MSELoss() if cfg.model == 'scae' else nn.CrossEntropyLoss()
    logger.info(f"Criterion: {criterion}")
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
