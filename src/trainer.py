import torch
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
from torch.optim import Optimizer
import torch.nn as nn
from typing import Optional
from torch.utils.data import DataLoader

LOG_STEP = 100


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: F,
        device: torch.device,
        train_loader: DataLoader,
        eval_loader: Optional[DataLoader] = None,
        save_path: Optional[str] = None,
    ):
        self._model = model.to(device)
        self._optimizer = optimizer
        self._criterion = criterion
        self._device = device
        self._train_loader = train_loader
        self._eval_loader = eval_loader
        self._save_path = save_path
        self._log_step = LOG_STEP

    def _train(self, epochs: int):
        raise NotImplementedError

    def _train_epoch(self):
        raise NotImplementedError

    def _save_weights(self):
        torch.save(self._model.state_dict(), self._save_path)
        print("Saved model weights at {}".format(self._save_path))

    # @property
    # def model(self) -> nn.Module:
    #     return self._model

    # @property
    # def optimzer(self) -> Optimizer:
    #     return self._optimizer

    # @property
    # def criterion(self) -> F:
    #     return self._criterion

    # @property
    # def device(self) -> torch.device:
    #     return self._device


class SCAETrainer(Trainer):
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: F,
        device: torch.device,
        train_loader: DataLoader,
        save_path: str,
    ):
        super().__init__(model, optimizer, criterion, device, train_loader, save_path)

    def _train(self, epochs: int):
        for epoch in range(epochs):
            loss_ = []
            for images in tqdm(self._train_loader):
                if isinstance(images, list):
                    # For VFR-syn-train dataset.
                    images = images[0]
                images = images.to(self._device)
                outputs = self._model(images)

                loss = self._criterion(outputs, images)

                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
                loss_.append(loss.item())

            print(f"Epoch [{epoch+1}/{epochs}], Loss: {np.mean(loss_):.4f}")


class SupervisedTrainer(Trainer):
    def __init__(
        self,
        model: nn.Module,
        optimizer: Optimizer,
        criterion: F,
        device: torch.device,
        train_loader: DataLoader,
        eval_loader: DataLoader,
        save_path: str,
        log_dir: str = './runs',
    ):
        super().__init__(
            model,
            optimizer,
            criterion,
            device,
            train_loader,
            eval_loader,
            save_path=save_path,
        )
        self._writer = SummaryWriter(log_dir=log_dir)

    def _train(self, epochs: int):
        best_acc = torch.inf
        for epoch in range(epochs):
            self._train_epoch(epoch)
            test_acc = self._evaluate(epoch)
            self._writer.add_scalar('Test Accuracy', test_acc, epoch)
            print(f"Test accuracy: {test_acc:.4f} at epoch {epoch+1}.")
            if test_acc < best_acc:
                self._save_weights(epoch)
                best_acc = test_acc
            # if epoch % 10 == 0:
            # self._save_weights(epoch)

    def _train_epoch(self, epoch: int):
        self._model.train()
        pbar = tqdm(enumerate(self._train_loader), total=len(self._train_loader))
        for idx, (images, labels) in pbar:
            images, labels = images.to(self._device), labels.to(self._device)
            with torch.set_grad_enabled(True):
                outputs = self._model(images)
                loss = self._criterion(outputs, labels)
                self._optimizer.zero_grad()
                loss.backward()
                self._optimizer.step()
            pbar.set_description(f"Loss: {loss.item():.4f}")

            # Log the loss to TensorBoard
            self._writer.add_scalar(
                'Training Loss', loss.item(), epoch * len(self._train_loader) + idx
            )

            # Every 20 steps, log the gradient histograms
            if idx % self._log_step == 0:
                for name, param in self._model.named_parameters():
                    if param.grad is not None:
                        self._writer.add_histogram(
                            f"{name}.grad",
                            param.grad,
                            epoch * len(self._train_loader) + idx,
                        )

    def _evaluate(self, epoch: int) -> float:
        self._model.eval()
        total_loss = 0
        with torch.no_grad():
            pbar = tqdm(
                enumerate(
                    self._eval_loader,
                ),
                total=len(
                    self._eval_loader,
                ),
            )
            for idx, (images, labels) in pbar:
                images, labels = images.to(self._device), labels.to(self._device)
                outputs = self._model(images)
                loss = self._criterion(outputs, labels)
                total_loss += loss.item()

                # Log the evaluation loss to TensorBoard
                # self.writer.add_scalar('Evaluation Loss', loss.item(), epoch * len(eval_loader) + idx)

        avg_loss = total_loss / len(self._eval_loader)
        return avg_loss

    def _save_weights(self, epoch: int):
        model_name = self._model.modules.name if hasattr(
            self._model, 'modules'
        ) else self._model.name
        path = self._save_path + f'{model_name}_weights_{epoch}.pth'
        torch.save(self._model.state_dict(), path)
        print("Saved model weights at {}".format(path))
