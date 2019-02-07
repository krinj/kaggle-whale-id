# -*- coding: utf-8 -*-

"""
<ENTER DESCRIPTION HERE>
"""

import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet

__author__ = "Jakrin Juangbhanich"
__email__ = "juangbhanich.k@gmail.com"


class WhaleNet(nn.Module):

    def __init__(self, freeze: bool = False, nf: int = 4):

        super().__init__()

        self.bone = resnet.resnet18(True)

        if freeze:
            for param in self.bone.parameters():
                param.requires_grad = False

        self.ada = nn.AdaptiveAvgPool2d(5)
        self.head = nn.Sequential(
            nn.Linear(12800, 256 * nf),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256 * nf, 64 * nf),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64 * nf, 32 * nf)
        )

        self.bone.avgpool = self.ada
        self.bone.fc = self.head

    def forward(self, x):
        """ Pass in an image Variable (PIL Image, put the channels first).
        Returns:
            Variable: A torch variable of length 16, containing the embedding of this image.
        """
        x = self.bone.forward(x)
        output = F.normalize(x, p=2, dim=1)
        return output
