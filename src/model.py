import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch import Tensor
from torch.optim import Optimizer, AdamW
from .backbone.resnet import ResNet, Bottleneck, BasicBlock
from .config import VFR_FONTS_NUM
import logging

logger = logging.getLogger(__name__)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=58)
        self.batch_norm = nn.BatchNorm2d(64)
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

    def forward(self, X: Tensor) -> Tensor:
        X = F.relu(self.conv1(X))  # output shape: 64 * 48 * 48
        X = self.batch_norm(X)
        X = self.max_pool(X)  # output shape: 64 * 24 * 24
        X = F.relu(self.conv2(X))  # output shape: 128 * 24 * 24

        return X


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.deconv1 = nn.ConvTranspose2d(128, 64, kernel_size=3, padding=1)
        self.unpool = nn.UpsamplingNearest2d(scale_factor=2)
        self.batch_norm = nn.BatchNorm2d(64)
        self.deconv2 = nn.ConvTranspose2d(64, 1, kernel_size=58)

    def forward(self, X: Tensor) -> Tensor:
        X = F.relu(self.deconv1(X))  # output shape: 64 * 24 * 24
        X = self.batch_norm(X)
        X = self.unpool(X)  # output shape: 64 * 48 * 48
        X = F.sigmoid(self.deconv2(X))  # output shape: 1 * 105 * 105

        return X


class SCAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def forward(self, X: Tensor) -> Tensor:
        X = self.encoder(X)
        X = self.decoder(X)

        return X

    def _load_weights(self, path: str) -> None:
        # Load the state_dict
        state_dict = torch.load(path)

        # Check if the model was parallelized during training
        if 'module' in list(state_dict.keys())[0]:
            # If it was parallelized, we need to remove the 'module.' prefix from keys
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        self.load_state_dict(state_dict)


class CNN(nn.Module):
    def __init__(
        self,
        num_classes: int = VFR_FONTS_NUM,
        encoder_weight_path: Optional[str] = None,
        finetune_ratio: Optional[float] = 0.0,
    ):
        super().__init__()
        # self.cls_flg = True if num_types is not None else False
        self._num_classes = num_classes
        self._scae = SCAE()
        self._use_SCAE = True if encoder_weight_path is not None else False
        self._finetune_ratio = finetune_ratio
        if self._use_SCAE:
            self._scae._load_weights(encoder_weight_path)
            logger.info("Using SCAE weights")
            self.Cu = nn.Sequential(
                self._scae.encoder.conv1,
                nn.ReLU(),  # output shape: 64 * 48 * 48
                nn.BatchNorm2d(64),  # output shape: 64 * 48 * 48
                nn.MaxPool2d(kernel_size=2),  # output shape: 64 * 24 * 24
                self._scae.encoder.conv2,
                nn.ReLU(),  # output shape: 128 * 24 * 24
                nn.BatchNorm2d(128),  # output shape: 128 * 24 * 24
                nn.MaxPool2d(kernel_size=2),  # output shape: 128 * 12 * 12
            )
        else:
            logger.info("Don't transfer from SCAE.")
            self.Cu = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=48, stride=1),
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=24, stride=1),
                nn.ReLU(),
                nn.BatchNorm2d(128),
                nn.ConvTranspose2d(
                    128, 128, kernel_size=3, stride=2, padding=1, output_padding=1
                ),
            )

        self.Cs = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),  # output shape: 256 * 12 * 12
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),  # output shape: 256 * 12 * 12
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(),  # output shape: 256 * 12 * 12
            nn.Flatten(),  # output shape: 36864
            nn.Linear(12 * 12 * 256, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            # for AdobeVFR syn datasets.
            nn.Linear(4096, VFR_FONTS_NUM),
            nn.ReLU(),
        )
        self.cls_head = nn.Linear(VFR_FONTS_NUM, self._num_classes)

    def forward(self, X: Tensor) -> Tensor:
        X = self.Cu(X)
        X = self.Cs(X)
        if self._num_classes != VFR_FONTS_NUM:
            X = self.cls_head(X)
        return X

    def _load_weights(self, path) -> None:
        # Load the state_dict
        state_dict = torch.load(path)

        # Check if the model was parallelized during training
        if 'module' in list(state_dict.keys())[0]:
            # If it was parallelized, we need to remove the 'module.' prefix from keys
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        self.load_state_dict(state_dict)

    def _optim_groups(self, lr: float) -> list:
        layers = {
            'cu': self.Cu,
            'cs': self.Cs,
        }

        optim_groups = []
        freeze_list = ['cu']
        for layer_name, layer in layers.items():
            current_lr = lr * self._finetune_ratio if layer_name in freeze_list else lr
            optim_groups.append({'params': layer.parameters(), 'lr': current_lr})
            logger.info(f"{layer_name}: {current_lr}")
        return optim_groups

    @property
    def name(self) -> str:
        return 'CNN'

    @property
    def num_classes(self) -> int:
        return self._num_classes


class FontResNet(nn.Module):
    def __init__(
        self,
        num_classes: int = VFR_FONTS_NUM,
        backbone: str = 'resnet50',
        finetune_ratio: Optional[float] = 0.0,
        pretrained_weight_path: Optional[str] = None,
    ):
        super(FontResNet, self).__init__()
        self._num_classes = num_classes
        self._finetune_ratio = finetune_ratio
        if backbone == 'resnet50':
            # Load the ResNet-50 model for grey-scale
            resnet_model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes)
        elif backbone == 'resnet18':
            # Load the ResNet-18 model for grey-scale
            resnet_model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)
        else:
            raise ValueError('Invalid backbone')
        # Separate out the ResNet components
        self.conv1 = resnet_model.conv1
        self.bn1 = resnet_model.bn1
        self.relu = resnet_model.relu
        self.maxpool = resnet_model.maxpool
        self.layer1 = resnet_model.layer1
        self.layer2 = resnet_model.layer2
        self.layer3 = resnet_model.layer3
        self.layer4 = resnet_model.layer4
        self.avgpool = resnet_model.avgpool

        # Classification layer
        self.fc = nn.Linear(resnet_model.fc.in_features, self._num_classes)
        
        if pretrained_weight_path is not None:
            self._load_weights(pretrained_weight_path)

    def forward(self, X: Tensor) -> Tensor:
        X = self.conv1(X)
        X = self.bn1(X)
        X = self.relu(X)
        X = self.maxpool(X)
        X = self.layer1(X)
        X = self.layer2(X)
        X = self.layer3(X)
        X = self.layer4(X)
        X = self.avgpool(X)
        X = torch.flatten(X, 1)
        X = self.fc(X)
        return X

    def _get_optimizer(self, lr: float, weigt_decay: float) -> Optimizer:
        return AdamW(self.parameters(), lr=lr, weight_decay=weigt_decay)

    def _load_weights(self, path: str) -> None:
        # Load the state_dict
        state_dict = torch.load(path)

        # Check if the model was parallelized during training
        if 'module' in list(state_dict.keys())[0]:
            # If it was parallelized, we need to remove the 'module.' prefix from keys
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        self.load_state_dict(state_dict)
        logger.info(f"<{self.name}> Loaded weights from {path}")

    def _optim_groups(self, lr: float) -> list:
        layers = {
            'conv1': self.conv1,
            'bn1': self.bn1,
            'layer1': self.layer1,
            'layer2': self.layer2,
            'layer3': self.layer3,
            'layer4': self.layer4,
            'avgpool': self.avgpool,
            'fc': self.fc,
        }

        optim_groups = []
        freeze_list = ['conv1', 'bn1', 'layer1', 'layer2', 'layer3']
        # no_finetune_list = ['layer4', 'avgpool', 'fc']
        for layer_name, layer in layers.items():
            current_lr = lr * self._finetune_ratio if layer_name in freeze_list else lr
            optim_groups.append({'params': layer.parameters(), 'lr': current_lr})
            logger.info(f"{layer_name}: {current_lr}")

        return optim_groups

    
    def _extract_embedding(self, X: Tensor) -> Tensor:
        X = self.conv1(X)
        X = self.bn1(X)
        X = self.relu(X)
        X = self.maxpool(X)
        X = self.layer1(X)
        X = self.layer2(X)
        X = self.layer3(X)
        X = self.layer4(X)
        X = self.avgpool(X)
        return torch.flatten(X, 1)

    @property
    def name(self) -> str:
        return 'ResNet'

    @property
    def num_classes(self) -> int:
        return self._num_classes
