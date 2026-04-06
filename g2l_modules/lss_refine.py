# models/lss_refine.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class LSSRefine(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.hss = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.GELU(), nn.Conv2d(dim, dim, 1)
        )
        self.dw5 = nn.Sequential(
            nn.Conv2d(dim, dim, 5, padding=2, groups=dim),
            nn.GELU(), nn.Conv2d(dim, dim, 1)
        )
        self.dw7 = nn.Sequential(
            nn.Conv2d(dim, dim, 7, padding=3, groups=dim),
            nn.GELU(), nn.Conv2d(dim, dim, 1)
        )
        self.fuse = nn.Conv2d(dim*3, dim, 1)

    def forward(self, f):
        g  = self.hss(f)
        l5 = self.dw5(f)
        l7 = self.dw7(f)
        return self.fuse(torch.cat([g, l5, l7], dim=1))

def upsample_and_smooth(amap_patch, img_hw):
    # amap_patch: [B,1,H,W]（patch 网格级）
    out = F.interpolate(amap_patch, size=img_hw, mode='bilinear', align_corners=False)
    # 轻量导向平滑（保边抑噪）
    edge = torch.nn.AvgPool2d(3,1,1)(out) - out
    out  = (out - 0.3 * edge).clamp_min(0.)
    return out
