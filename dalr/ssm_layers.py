# Dinomaly/models_mambaAD/mamba_layers.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class SelectiveSSM(nn.Module):
    """
    简化工程实现：线性复杂度的“选择性状态空间”层
    输入:  x [B, L, C]
    输出:  y [B, L, C]
    """
    def __init__(self, dim, d_state=16, dt_rank=32, dropout=0.0):
        super().__init__()
        self.in_proj  = nn.Linear(dim, dim * 2)            # produce a, g
        self.dwconv   = nn.Conv1d(dim, dim, 3, 1, 1, groups=dim)  # depthwise conv ~ selective scan
        self.out_proj = nn.Linear(dim, dim)
        self.dropout  = nn.Dropout(dropout)

    def forward(self, x):        # x: [B, L, C]
        a, g = self.in_proj(x).chunk(2, dim=-1)
        y = torch.relu(a)
        y = y.transpose(1, 2)    # [B, C, L]
        y = self.dwconv(y)
        y = y.transpose(1, 2)    # [B, L, C]
        y = y * torch.sigmoid(g) # selective gating
        y = self.out_proj(y)
        return self.dropout(y)
