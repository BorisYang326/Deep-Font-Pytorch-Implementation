import torch
from tqdm import tqdm
import numpy as np
from torch.utils.tensorboard import SummaryWriter

class Trainer:
    def __init__(self, model, optimizer, criterion, device,save_path=None):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.save_path = save_path

    def _train(self, train_loader, epochs):
        raise NotImplementedError

    def _train_epoch(self, train_loader):
        raise NotImplementedError
    
    def _save_weights(self):
        torch.save(self.model.state_dict(), self.save_path)
        print("Saved model weights at {}".format(self.save_path))
    
    def _get_model(self):
        return self.model


class SCAETrainer(Trainer):
    def __init__(self, model, optimizer, criterion, device, save_path=None):
        super().__init__(model, optimizer, criterion, device, save_path)
    def _train(self, train_loader, epochs):
        for epoch in range(epochs):
            loss_ = []
            for images in tqdm(train_loader):
                if isinstance(images, list):
                    # For VFR-syn-train dataset.
                    images = images[0]
                images = images.to(self.device)
                outputs = self.model(images)
                
                loss = self.criterion(outputs, images)
                
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                loss_.append(loss.item())
                
            print(f"Epoch [{epoch+1}/{epochs}], Loss: {np.mean(loss_):.4f}")


class CNNTrainer(Trainer):
    def __init__(self, model, optimizer, criterion, device, eval_loader, save_path=None, log_dir='./runs'):
        super().__init__(model, optimizer, criterion, device, save_path=save_path)
        self.eval_loader = eval_loader
        self.writer = SummaryWriter(log_dir=log_dir)

    def _train(self, train_loader, epochs):
        best_acc = torch.inf  
        for epoch in range(epochs):
            self._train_epoch(train_loader, epoch)
            test_acc = self._evaluate(self.eval_loader, epoch)
            self.writer.add_scalar('Test Accuracy', test_acc, epoch)
            print(f"Test accuracy: {test_acc:.4f} at epoch {epoch+1}.")
            if test_acc < best_acc:
                self._save_weights()
                best_acc = test_acc
            if epoch % 10 == 0:
                self._save_weights(epoch)

    def _train_epoch(self, train_loader, epoch):
        self.model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for idx, (images, labels) in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            with torch.set_grad_enabled(True):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            pbar.set_description(f"Loss: {loss.item():.4f}")
            
            # Log the loss to TensorBoard
            self.writer.add_scalar('Training Loss', loss.item(), epoch * len(train_loader) + idx)
            
            # Every 20 steps, log the gradient histograms
            if idx % 20 == 0:
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        self.writer.add_histogram(f"{name}.grad", param.grad, epoch * len(train_loader) + idx)


    def _evaluate(self, eval_loader, epoch):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            pbar = tqdm(enumerate(eval_loader), total=len(eval_loader))
            for idx, (images, labels) in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                # Log the evaluation loss to TensorBoard
                # self.writer.add_scalar('Evaluation Loss', loss.item(), epoch * len(eval_loader) + idx)
                
        avg_loss = total_loss / len(eval_loader)
        return avg_loss
    
    def _save_weights(self,epoch):
        path = self.save_path + f'cnn_weights_{epoch}.pth'
        torch.save(self.model.state_dict(), path)
        print("Saved model weights at {}".format(path))

    
class FontResNetTrainer(Trainer):
    def __init__(self, model, optimizer, criterion, device, eval_loader, save_path=None, log_dir='./runs/resnet'):
        super().__init__(model, optimizer, criterion, device, save_path=save_path)
        # self.eval_loader = eval_loader
        self.writer = SummaryWriter(log_dir=log_dir)

    def _train(self, train_loader, epochs):
        # best_acc = torch.inf  
        for epoch in range(epochs):
            self._train_epoch(train_loader, epoch)
            # test_acc = self._evaluate(self.eval_loader, epoch)
            # self.writer.add_scalar('Test Accuracy', test_acc, epoch)
            # print(f"Test accuracy: {test_acc:.4f} at epoch {epoch+1}.")
            # if test_acc < best_acc:
            #     self._save_weights()
            #     best_acc = test_acc
            if epoch % 5 == 0:
                self._save_weights(epoch)
            

    def _train_epoch(self, train_loader, epoch):
        self.model.train()
        pbar = tqdm(enumerate(train_loader), total=len(train_loader))
        for idx, (images, labels) in pbar:
            images, labels = images.to(self.device), labels.to(self.device)
            with torch.set_grad_enabled(True):
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            pbar.set_description(f"Loss: {loss.item():.4f}")
            
            # Log the loss to TensorBoard
            self.writer.add_scalar('Training Loss', loss.item(), epoch * len(train_loader) + idx)
            
            # Every 20 steps, log the gradient histograms
            if idx % 20 == 0:
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        self.writer.add_histogram(f"{name}.grad", param.grad, epoch * len(train_loader) + idx)


    def _evaluate(self, eval_loader, epoch):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            pbar = tqdm(enumerate(eval_loader), total=len(eval_loader))
            for idx, (images, labels) in pbar:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()
                
                # Log the evaluation loss to TensorBoard
                # self.writer.add_scalar('Evaluation Loss', loss.item(), epoch * len(eval_loader) + idx)
                
        avg_loss = total_loss / len(eval_loader)
        return avg_loss
    
    def _save_weights(self,epoch):
        path = self.save_path + f'resnet_weights_{epoch}.pth'
        torch.save(self.model.state_dict(), path)
        print("Saved model weights at {}".format(path))