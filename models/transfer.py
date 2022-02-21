import pretrainedmodels
import torch.nn.functional as F
import torch.nn as nn
import torchvision.models as models
import torch

class ResNet50(nn.Module):
    def __init__(self, pretrained):
        super(ResNet50, self).__init__()
        if pretrained is True:
            self.model = pretrainedmodels.__dict__['resnet50'](pretrained='imagenet')
        else:
            self.model = pretrainedmodels.__dict__['resnet50'](pretrained=None)
        
        self.l0 = nn.Linear(2048, 101)
        self.dropout = nn.Dropout2d(0.4)

    def forward(self, x):
        # get the batch size only, ignore (c, h, w)
        batch, _, _, _ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        x = self.dropout(x)
        l0 = self.l0(x)
        return l0

class LowRankResNet50(nn.Module):
    def __init__(self, pretrained):
        super(LowRankResNet50, self).__init__()
        if pretrained is True:
            self.model = pretrainedmodels.__dict__['resnet50'](pretrained='imagenet')
        else:
            self.model = pretrainedmodels.__dict__['resnet50'](pretrained=None)
        
        self.l0 = nn.Linear(2048, 101)
        self.dropout = nn.Dropout2d(0.4)

    def forward(self, x):
        # get the batch size only, ignore (c, h, w)
        batch, _, _, _ = x.shape
        x = self.model.features(x)
        x = F.adaptive_avg_pool2d(x, 1).reshape(batch, -1)
        x = self.dropout(x)
        l0 = self.l0(x)
        return l0