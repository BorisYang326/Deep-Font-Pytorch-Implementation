import torch
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from torch.nn import functional as F
from torch.optim import Optimizer
import torch.nn as nn
from typing import Optional
from torch.utils.data import DataLoader
import os
from collections import defaultdict
from typing import Tuple, Dict
import pickle


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
        log_dir: Optional[str] = None,
        log_step: Optional[int] = None,
    ):
        self._model = model.to(device)
        self._optimizer = optimizer
        self._criterion = criterion
        self._device = device
        self._train_loader = train_loader
        self._eval_loader = eval_loader
        self._save_path = save_path
        self._log_step = log_step
        self._writer = SummaryWriter(log_dir=log_dir)

    def _train(self, epochs: int):
        raise NotImplementedError

    def _train_epoch(self):
        raise NotImplementedError

    def _save_weights(self):
        os.makedirs(self._save_path, exist_ok=True)
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
        eval_loader: Optional[DataLoader] = None,
        log_dir: Optional[str] = None,
    ):
        super().__init__(
            model,
            optimizer,
            criterion,
            device,
            train_loader,
            eval_loader,
            save_path,
            log_dir,
        )

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
        log_step: Optional[int] = None,
    ):
        super().__init__(
            model,
            optimizer,
            criterion,
            device,
            train_loader,
            eval_loader,
            save_path,
            log_dir,
            log_step,
        )
        self._model_name = model.name
        if torch.cuda.device_count() > 1:
            self._model = nn.DataParallel(model)
        else:
            self._model = model

    def _train(self, epochs: int):
        best_acc = 0.0
        for epoch in range(epochs):
            self._train_epoch(epoch)
            test_loss, test_acc, class_acc_dic = self._evaluate(epoch)
            self._writer.add_scalar('Test Accuracy', test_acc, epoch)
            self._writer.add_scalar('Test Loss', test_loss, epoch)
            # self._writer.add_histogram(
            #     'Class_Accuracy_Distribution',
            #     torch.tensor(list(class_acc_dic.values())).view(2383, 1),
            #     epoch,
            # )
            print(f"Test accuracy: {test_acc:.4f} at epoch {epoch+1}.")
            print(f"Test loss: {test_loss:.4f} at epoch {epoch+1}.")
            if test_acc > best_acc:
                self._save_weights(epoch)
                # save class accuracy
                with open(os.path.join(self._save_path, 'class_accuracy.pkl'), 'wb') as handle:
                    pickle.dump(class_acc_dic, handle, protocol=pickle.HIGHEST_PROTOCOL)
                best_acc = test_acc
            # if epoch % 10 == 0:
            #     self._save_weights(epoch)

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

    def _evaluate(self, epoch: int) -> Tuple[float, float, Dict[int, float]]:
        self._model.eval()
        total_loss = 0
        total_accuracy = []
        correct_counts = defaultdict(int)  # Correct predictions for each class
        total_counts = defaultdict(int)  # Total predictions for each class
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
                predicted_labels = outputs.argmax(dim=1)
                total_loss += loss.item()

                for label in labels.unique():
                    correct_counts[label.item()] += (
                        (predicted_labels[labels == label] == label).sum().item()
                    )
                    total_counts[label.item()] += (labels == label).sum().item()

                accuracy = (predicted_labels == labels).float().mean()
                total_accuracy.append(accuracy.item())

                # Log the evaluation loss to TensorBoard
                # self.writer.add_scalar('Evaluation Loss', loss.item(), epoch * len(eval_loader) + idx)

        avg_loss = total_loss / len(self._eval_loader)
        avg_accuracy = np.mean(total_accuracy)
        class_accuracies = {
            label: correct_counts[label] / total_counts[label] for label in total_counts
        }
        return avg_loss, avg_accuracy, class_accuracies

    def _save_weights(self, epoch: int):
        os.makedirs(self._save_path, exist_ok=True)
        path = os.path.join(self._save_path, f'{self._model_name}_weights_{epoch}.pth')
        torch.save(self._model.state_dict(), path)
        print("Saved model weights at {}".format(path))