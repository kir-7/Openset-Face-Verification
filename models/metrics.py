import torch
import numpy as np
from torch import nn

from torch.nn import functonal as F 

import math

from train import device


class ArcFace(torch.nn.Module):
    def __init__(self, n_classes, emb_dim, s=30.0, m=0.5, easy_margin=False):
        super().__init__()

        self.n_classes = n_classes
        self.emb_dim = emb_dim

        self.s = s
        self.m = m

        self.weight = torch.nn.Parameter(torch.FloatTensor(self.n_classes, emb_dim)).to(device)
        torch.nn.init.kaiming_uniform_(self.weight) # weight init


        self.easy_margin= easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi -m)*m



    def forward(self, x, label):

        cosine = F.linear(F.normalize(x), F.normalize(self.weight))
        sine = torch.sqrt(1-torch.pow(cosine, 2)).clamp(0, 1)
        phi = cosine*self.cos_m - sine*self.sin_m
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
        output *= self.s

        return output