import random

import albumentations
import torch
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
import sys
sys.path.append(sys.path[0].replace('networks', ''))
from data.flamby_fed_isic2019 import FedIsic2019

class MyEfficientNet(nn.Module):
    """Baseline model
    We use the EfficientNets architecture that many participants in the ISIC
    competition have identified to work best.
    See here the [reference paper](https://arxiv.org/abs/1905.11946)
    Thank you to [Luke Melas-Kyriazi](https://github.com/lukemelas) for his
    [pytorch reimplementation of EfficientNets]
    (https://github.com/lukemelas/EfficientNet-PyTorch).
    """

    def __init__(self, pretrained=True, arch_name="efficientnet-b0", num_classes=8):
        super(MyEfficientNet, self).__init__()
        self.pretrained = pretrained
        self.base_model = (
            EfficientNet.from_pretrained(arch_name)
            if pretrained
            else EfficientNet.from_name(arch_name)
        )
        # self.base_model=torchvision.models.efficientnet_v2_s(pretrained=pretrained)
        nftrs = self.base_model._fc.in_features
        # print("Number of features output by EfficientNet", nftrs)
        self.base_model._fc = nn.Linear(nftrs, num_classes)

    def forward(self, image, feature_out=False):
        # Convolution layers
        x = self.base_model.extract_features(image)
        # Pooling and final linear layer
        feature_x = self.base_model._avg_pooling(x)
        if self.base_model._global_params.include_top:
            x = feature_x.flatten(start_dim=1)
            x = self.base_model._dropout(x)
            x = self.base_model._fc(x)
        if feature_out:
            return x, feature_x.flatten(start_dim=1)
        return x


