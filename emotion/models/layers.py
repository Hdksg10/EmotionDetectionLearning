
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )
        # nn.init.kaiming_normal_(self.fc[0].weight)
        # nn.init.kaiming_normal_(self.fc[2].weight)
        

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)
    
class SimAM(nn.Module):
        # X: input feature [N, C, H, W]
        # lambda: coefficient λ in Eqn (5)
        def forward(self, x, param = 1e-4):
            # print(x.shape)
            # spatial size
            n = x.shape[2] * x.shape[3] - 1
            # square of (t - u)
            d = (x - x.mean(dim=[2,3], keepdim=True)).pow(2)
            # d.sum() / n is channel variance
            v = d.sum(dim=[2,3], keepdim=True) / n
            # E_inv groups all importance of X
            E_inv = d / (4 * (v + param)) + 0.5
            # return attended features
            return x * torch.sigmoid(E_inv)
        
class STN(nn.Module):
    def __init__(self):
        super(STN, self).__init__()
        self.localization = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=7),
            nn.BatchNorm2d(8),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
            nn.Conv2d(8, 10, kernel_size=5),
            nn.BatchNorm2d(10),
            nn.MaxPool2d(2, stride=2),
            nn.ReLU(True),
        )
    
        self.fc_loc = nn.Sequential(
            nn.Linear(10 * 52 * 52, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2),
        )
        
    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 52 * 52)
        theta = self.fc_loc(xs)
        theta = theta.view(-1, 2, 3)
        
        grid = F.affine_grid(theta, x.size(), align_corners=True)
        x = F.grid_sample(x, grid, align_corners=True)
        return x
