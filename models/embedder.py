# models/embedder.py
import torch
import torch.nn as nn
import torchvision.models as tv

class Embedder(nn.Module):
    def __init__(self, out_dim=128):
        super().__init__()
        base = tv.resnet18(pretrained=False)
        base.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        self.feats = nn.Sequential(*list(base.children())[:-1])
        self.fc = nn.Linear(base.fc.in_features, out_dim)

    def forward(self, x):
        f = self.feats(x)
        f = f.view(f.size(0), -1)
        f = self.fc(f)
        return f / (f.norm(dim=1, keepdim=True) + 1e-9)
