import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch import Tensor
from torch.optim import Optimizer,AdamW
from src.backbone.resnet import ResNet, Bottleneck,BasicBlock

INPUT_SIZE = 105
USE_SCAE_WEIGHTS = True


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=58)
        self.batch_norm = nn.BatchNorm2d(64)
        self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

    def forward(self, X:Tensor)->Tensor:
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

    def forward(self, X:Tensor)->Tensor:
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

    def forward(self, X:Tensor)->Tensor:
        X = self.encoder(X)
        X = self.decoder(X)

        return X

    def _load_weights(self, path:str)->None:
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
        encoder: Optional[Encoder] = None,
        num_types: Optional[int] = None,
        finetune_ratio_Cu: Optional[float] = 0.0,
    ):
        super().__init__()
        self.cls_flg = True if num_types is not None else False
        self.use_SCAE = True if encoder is not None else False
        self.finetune_ratio_Cu = finetune_ratio_Cu
        if self.use_SCAE:
            print("Using SCAE weights")
            self.Cu = nn.Sequential(
                encoder.conv1,
                nn.ReLU(),  # output shape: 64 * 48 * 48
                nn.BatchNorm2d(64),  # output shape: 64 * 48 * 48
                nn.MaxPool2d(kernel_size=2),  # output shape: 64 * 24 * 24
                encoder.conv2,
                nn.ReLU(),  # output shape: 128 * 24 * 24
                nn.BatchNorm2d(128),  # output shape: 128 * 24 * 24
                nn.MaxPool2d(kernel_size=2),  # output shape: 128 * 12 * 12
            )
        else:
            print("Don't transfer from SCAE.")
            self.Cu = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=48, stride=1),
                nn.ReLU(),
                nn.BatchNorm2d(64),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 128, kernel_size=24, stride=1),
                nn.ReLU(),
                nn.BatchNorm2d(128),
                nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1),
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
            nn.Linear(4096, 2383),
            nn.ReLU(),
        )
        if self.cls_flg:
            self.cls_head = nn.Linear(2383, num_types)
        else:
            self.cls_head = nn.Identity()

    def forward(self, X:Tensor)->Tensor:
        X = self.Cu(X)
        X = self.Cs(X)
        X = self.cls_head(X)
        return X

    def _load_weights(self, path)->None:
        # Load the state_dict
        state_dict = torch.load(path)

        # Check if the model was parallelized during training
        if 'module' in list(state_dict.keys())[0]:
            # If it was parallelized, we need to remove the 'module.' prefix from keys
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        self.load_state_dict(state_dict)

    def _get_optimizer(self, lr: float)->Optimizer:
        cnn_optimizer = AdamW(
            [
                {'params': self.Cu.parameters(), 'lr': lr * self.finetune_ratio_Cu},
                {'params': self.Cs.parameters(), 'lr': lr},
            ],
        )
        return cnn_optimizer
    
    @property
    def name(self)->str:
        return 'CNN'


class FontResNet(nn.Module):
    def __init__(self, num_classes=2383):
        super(FontResNet, self).__init__()
        
        # Load the ResNet-50 model for grey-scale
        # self.resnet_ = ResNet(Bottleneck, [3, 4, 6, 3])
        # Load the ResNet-18 model for grey-scale
        self.resnet_ = ResNet(BasicBlock, [2, 2, 2, 2])
        
        # Remove the last fully connected layer to get the embeddings
        modules = list(self.resnet_.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        
        # Embedding layer
        # self.embedding = nn.Linear(self.resnet.fc.in_features, 512)
        
        # Classification layer
        self.fc = nn.Linear(self.resnet_.fc.in_features, num_classes)
        
    def forward(self, X:Tensor)->Tensor:
        X = self.resnet(X)
        X = torch.flatten(X, 1)
        # X = self.embedding(X)
        X = self.fc(X)
        return F.softmax(X, dim=1)
    
    def _get_optimizer(self, lr: float)->Optimizer:
        return AdamW(self.parameters(), lr=lr)
    
    
    def _load_weights(self, path:str)->None:
        # Load the state_dict
        state_dict = torch.load(path)

        # Check if the model was parallelized during training
        if 'module' in list(state_dict.keys())[0]:
            # If it was parallelized, we need to remove the 'module.' prefix from keys
            state_dict = {k[7:]: v for k, v in state_dict.items()}
        self.load_state_dict(state_dict)
        
    @property
    def name(self)->str:
        return 'ResNet'