# models/pairnet.py
import torch
import torch.nn as nn
import torchvision.models as tv

class PairNet(nn.Module):
    def __init__(self):
        super().__init__()
        base = tv.resnet18(pretrained=False)
        base.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        self.backbone = nn.Sequential(*list(base.children())[:-2])
        self.pool = nn.AdaptiveAvgPool2d(1)
        feat_dim = list(base.children())[-1].in_features
        self.fc = nn.Sequential(
            nn.Linear(feat_dim*2, 512), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(512, 1), nn.Sigmoid()
        )

    def forward(self, a, b):
        fa = self.pool(self.backbone(a)).view(a.size(0), -1)
        fb = self.pool(self.backbone(b)).view(b.size(0), -1)
        return self.fc(torch.cat([fa, fb], 1)).squeeze(1)
