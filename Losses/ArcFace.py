import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class ArcFace(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.5):
        super().__init__()
        self.W = nn.Parameter(torch.randn(out_features, in_features))
        nn.init.xavier_uniform_(self.W)

        self.s = s
        self.m = m
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, x, labels):
        x = F.normalize(x)
        W = F.normalize(self.W)

        # cos(theta)
        cos_theta = torch.matmul(x, W.T)

        # sin(theta)
        sin_theta = torch.sqrt(torch.clamp(1.0 - cos_theta ** 2, min=1e-7))

        # cos(theta + m)
        cos_theta_m = cos_theta * self.cos_m - sin_theta * self.sin_m

        # fallback (IMPORTANT)
        cos_theta_m = torch.where(
            cos_theta > self.th,
            cos_theta_m,
            cos_theta - self.mm
        )

        # one-hot replace
        one_hot = F.one_hot(labels, num_classes=W.shape[0]).float()
        output = one_hot * cos_theta_m + (1.0 - one_hot) * cos_theta

        logits = self.s * output

        return logits