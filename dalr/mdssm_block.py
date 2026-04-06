# # Dinomaly/models_mambaAD/hss_block.py
# import torch
# import torch.nn as nn
# from .hilbert_scan import scan_2d_to_1d, descan_1d_to_2d
# from .ssm_layers import SelectiveSSM

# _DEFAULT_DIRS = ('fwd','rev','wh_fwd','wh_rev','rot90_fwd','rot90_rev','wh_rot90_fwd','wh_rot90_rev')

# class MDSSMBlock(nn.Module):
#     def __init__(self, dim, scan_method='hilbert', directions=_DEFAULT_DIRS):
#         super().__init__()
#         self.norm1 = nn.LayerNorm(dim)
#         self.lin1  = nn.Linear(dim, dim)
#         self.dw3   = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
#         self.act   = nn.SiLU()
#         self.ssm   = SelectiveSSM(dim)
#         self.norm2 = nn.LayerNorm(dim)
#         self.lin2  = nn.Linear(dim, dim)
#         self.scan_method = scan_method
#         self.directions  = directions

#     def forward(self, x):
#         """
#         x: [B, C, H, W]
#         """
#         B, C, H, W = x.shape
#         xn = self.norm1(x.permute(0,2,3,1))  # [B,H,W,C]
#         y = self.lin1(xn).permute(0,3,1,2)   # [B,C,H,W]
#         y = self.dw3(y)
#         y = self.act(y)

#         agg = 0
#         for d in self.directions:
#             seq = scan_2d_to_1d(y, method=self.scan_method, direction=d)   # [B,L,C]
#             seq = self.ssm(seq)
#             feat = descan_1d_to_2d(seq, H, W, method=self.scan_method, direction=d)
#             agg = agg + feat
#         y = agg / len(self.directions)

#         gate = torch.sigmoid(self.lin2(self.norm2(x.permute(0,2,3,1))))  # [B,H,W,C]
#         y = y + (x * gate.permute(0,3,1,2))
#         return y

# Dinomaly/models_mambaAD/hss_block.py
import torch
import torch.nn as nn
from .hilbert_scan import scan_2d_to_1d, descan_1d_to_2d
from .ssm_layers import SelectiveSSM

_DEFAULT_DIRS = ('fwd','rev','wh_fwd','wh_rev','rot90_fwd','rot90_rev','wh_rot90_fwd','wh_rot90_rev')

class MDSSMBlock(nn.Module):
    def __init__(self, dim, scan_method='hilbert', directions=_DEFAULT_DIRS,
                 learn_dir=True, use_conv_gate=True):
        """
        learn_dir: 是否学习各扫描方向的权重（softmax），替代原来的等权平均
        use_conv_gate: 是否使用基于 [x,y] 的卷积门控，替代 LN->Linear 的纯 x 门控
        """
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.lin1  = nn.Linear(dim, dim)
        self.dw3   = nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim)
        self.act   = nn.SiLU()
        self.ssm   = SelectiveSSM(dim)

        # 原门控仍保留（开关可控），以便回退
        self.norm2 = nn.LayerNorm(dim)
        self.lin2  = nn.Linear(dim, dim)

        # 新增：卷积门控（基于 x 与 y）
        self.use_conv_gate = use_conv_gate
        if self.use_conv_gate:
            self.gate_conv = nn.Conv2d(dim * 2, dim, kernel_size=1)

        # 新增：方向权重（softmax）
        self.directions  = directions
        self.learn_dir   = learn_dir
        if self.learn_dir:
            self.dir_logits = nn.Parameter(torch.zeros(len(self.directions)))

        self.scan_method = scan_method

    def forward(self, x):
        """
        x: [B, C, H, W]
        """
        B, C, H, W = x.shape
        # 前置：LN + 1x1 + DWConv(3x3) + SiLU
        xn = self.norm1(x.permute(0,2,3,1))        # [B,H,W,C]
        y  = self.lin1(xn).permute(0,3,1,2)        # [B,C,H,W]
        y  = self.dw3(y)
        y  = self.act(y)

        # 多方向扫描 + 学习加权
        if self.learn_dir:
            weights = torch.softmax(self.dir_logits, dim=0)  # [D]
        agg = 0
        for i, d in enumerate(self.directions):
            seq  = scan_2d_to_1d(y, method=self.scan_method, direction=d)   # [B,L,C]
            seq  = self.ssm(seq)
            feat = descan_1d_to_2d(seq, H, W, method=self.scan_method, direction=d)  # [B,C,H,W]
            if self.learn_dir:
                agg = agg + weights[i] * feat
            else:
                agg = agg + feat
        if not self.learn_dir:
            agg = agg / len(self.directions)
        y = agg  # [B,C,H,W]

        # 门控残差：y = y + x + x * gate(x, y)
        if self.use_conv_gate:
            gate = torch.sigmoid(self.gate_conv(torch.cat([x, y], dim=1)))  # [B,C,H,W]
        else:
            gate = torch.sigmoid(self.lin2(self.norm2(x.permute(0,2,3,1)))).permute(0,3,1,2)
        y = y + x + x * gate
        return y
