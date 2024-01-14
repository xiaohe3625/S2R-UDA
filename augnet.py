import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class AugmentorRotation(nn.Module):
    def __init__(self):
        super(AugmentorRotation, self).__init__()
        self.conv1 = torch.nn.Conv1d(3, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 256, 1)
        self.conv3 = torch.nn.Conv1d(256, 1024, 1)
        self.conv4 = torch.nn.Conv1d(1024, 256, 1)
        self.conv5 = torch.nn.Conv1d(256, 64, 1)
        self.conv6 = torch.nn.Conv1d(64, 1, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(256)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(256)
        self.bn5 = nn.BatchNorm1d(64)
        self.conv7 = torch.nn.Conv1d(1024, 3, 1)
        self.scale_factors = nn.Parameter(torch.rand(3).add_(0.9))

    def forward(self, x):
        B, C, N = x.size()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        feat = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(feat)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.conv6(x)
        x = torch.max(x, 0, keepdim=False)[0].view(-1, 1)
        x = x * (1/16 * torch.pi)
        c, s = torch.cos(x), torch.sin(x)
        cs0 = torch.zeros_like(c)
        cs1 = torch.ones_like(c)
        R = torch.stack([c, -s, cs0, s, c, cs0, cs0, cs0, cs1], dim=-1).view(B, N, 3, 3)
        scale = 0.2 * torch.sigmoid(self.conv7(feat)) + 0.9
        scale = scale.view(-1, N, 3)
        return R, scale

class AugNet(nn.Module):
    def __init__(self, in_dim=3):
        super(AugNet, self).__init__()
        self.conv1 = torch.nn.Conv1d(in_dim, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 256, 1)
        self.conv4 = torch.nn.Conv1d(256, 1024, 1)
        self.conv5 = torch.nn.Conv1d(1024, 3, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(256)
        self.bn4 = nn.BatchNorm1d(1024)
        self.rot = AugmentorRotation()

    def forward(self, xyz):
        B, N, C = xyz.size()
        xyz = xyz.view(-1, 3, N)
        R, scale = self.rot(xyz)
        x = F.relu(self.bn1(self.conv1(xyz)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        noise = self.conv5(x)
        noise = noise
        p1 = random.uniform(-0.1, 0.1)
        noise = noise * p1
        noise = noise.view(-1, N, 3)
        return R, scale, noise


if __name__ == '__main__':
    xyz = torch.randn((1, 65536, 3))
    aug = AugNet()
    R, scale, noise = aug(xyz)
    print(R.shape)
