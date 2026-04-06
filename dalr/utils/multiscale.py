# Dinomaly/models_mambaAD/utils/multiscale.py
import torch
import torch.nn as nn

class HFPN(nn.Module):
    """
    轻量 Half-FPN：仅做 1x1 通道对齐（你可自行加 top-down 融合）
    """
    def __init__(self, in_dims, out_dims=(256, 512, 1024)):
        super().__init__()
        assert len(in_dims) == len(out_dims) == 3
        self.lateral = nn.ModuleList([nn.Conv2d(i, o, 1) for i, o in zip(in_dims, out_dims)])

    def forward(self, feats):
        # feats: [f1,f2,f3] (来自 encoder 的多层特征)
        outs = [l(f) for l, f in zip(self.lateral, feats)]
        return outs
