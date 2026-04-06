# models_mambaAD/pyramid_mamba_decoder.py
import math
from typing import Tuple, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# 直接从你现有模块导入（G2LHyD 中已使用该类）
from g2l_modules.g2l_model import DALRBlock


class _BlockStack(nn.Module):
    """对一串 DALRBlock 做顺序堆叠"""
    def __init__(self, embed_dim: int, depth: int, num_register_tokens: int, num_hss: int = 3):
        super().__init__()
        self.blocks = nn.ModuleList([
            DALRBlock(embed_dim=embed_dim,
                              num_register_tokens=num_register_tokens,
                              num_hss=num_hss)
            for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, L, C)
        for blk in self.blocks:
            x = blk(x)
        return x


class PyramidDecoder(nn.Module):
    """
    多尺度金字塔 Mamba 解码器（最小侵入式版本）
    - 基于 ViT patch tokens 的基尺度（一般 28x28）
    - 通过 2x / 4x 下采样构建 S2 / S3 分支
    - 各尺度分别走 Mamba 堆栈，然后上采样回基尺度并融合
    - 注册（register）tokens 只在基尺度参与（与 DINOv2 reg4 兼容）
    - 输出与输入 tokens 形状一致（包含 reg），可直接替换原解码结果
    """
    def __init__(
        self,
        embed_dim: int,
        num_register_tokens: int = 0,
        depth: int = 8,
        scales: Tuple[int, ...] = (1, 2, 4),
        fuse: str = "sum",                   # ['sum', 'concat']
        share_weights: bool = False,         # S2/S3 是否共享一套“无 reg”堆栈
        num_hss: int = 3
    ):
        super().__init__()
        assert 1 in scales, "scales 必须包含 1（基尺度）"
        self.embed_dim = embed_dim
        self.num_register_tokens = int(num_register_tokens)
        self.depth = int(depth)
        self.scales = tuple(sorted(set(scales)))
        self.fuse = fuse
        self.share_weights = bool(share_weights)

        # 基尺度：含 reg
        self.stack_s1 = _BlockStack(embed_dim, depth, num_register_tokens=self.num_register_tokens, num_hss=num_hss)

        # 下级尺度：不含 reg（序列只由 patch 组成）
        if any(s != 1 for s in self.scales):
            if self.share_weights:
                self.stack_small = _BlockStack(embed_dim, depth, num_register_tokens=0, num_hss=num_hss)
            else:
                if 2 in self.scales:
                    self.stack_s2 = _BlockStack(embed_dim, depth, num_register_tokens=0, num_hss=num_hss)
                if 4 in self.scales:
                    self.stack_s3 = _BlockStack(embed_dim, depth, num_register_tokens=0, num_hss=num_hss)

        # 融合投影（concat 时需要把 3C -> C）
        if self.fuse == "concat":
            n_branches = len(self.scales)
            self.fuse_proj = nn.Conv2d(in_channels=self.embed_dim * n_branches,
                                       out_channels=self.embed_dim,
                                       kernel_size=1,
                                       bias=True)
        elif self.fuse == "sum":
            self.fuse_proj = None
        else:
            raise ValueError("fuse must be 'sum' or 'concat'")

    # ---------- 基础变形工具 ----------
    @staticmethod
    def _split_reg(tokens: torch.Tensor, num_reg: int):
        # tokens: (B, L, C)
        if num_reg > 0:
            reg = tokens[:, :num_reg, :]
            patch = tokens[:, num_reg:, :]
        else:
            reg = None
            patch = tokens
        return reg, patch

    @staticmethod
    def _cat_reg(reg: Optional[torch.Tensor], patch: torch.Tensor):
        return torch.cat([reg, patch], dim=1) if reg is not None else patch

    @staticmethod
    def _tokens_to_grid(patch_tokens: torch.Tensor, hw: Tuple[int, int]) -> torch.Tensor:
        # patch_tokens: (B, H*W, C) -> (B, C, H, W)
        B, N, C = patch_tokens.shape
        H, W = hw
        assert H * W == N, f"tokens_to_grid: H*W({H}*{W}) != N({N})"
        return patch_tokens.view(B, H, W, C).permute(0, 3, 1, 2).contiguous()

    @staticmethod
    def _grid_to_tokens(grid: torch.Tensor) -> torch.Tensor:
        # grid: (B, C, H, W) -> (B, H*W, C)
        B, C, H, W = grid.shape
        return grid.permute(0, 2, 3, 1).contiguous().view(B, H * W, C)

    def _run_small_stack(self, tokens: torch.Tensor, scale: int) -> torch.Tensor:
        # tokens: (B, N, C), 不含 reg
        if self.share_weights:
            return self.stack_small(tokens)
        else:
            if scale == 2:
                return self.stack_s2(tokens)
            elif scale == 4:
                return self.stack_s3(tokens)
            else:
                raise RuntimeError("unexpected scale for small stack")

    # ---------- 前向 ----------
    def forward(self, tokens: torch.Tensor, patch_hw: Optional[Tuple[int, int]] = None) -> torch.Tensor:
        """
        tokens: (B, num_reg + H*W, C) 或 (B, H*W, C)
        patch_hw: (H, W). 若不给，将自动从长度推断为正方形网格
        """
        B, L, C = tokens.shape
        num_reg = self.num_register_tokens
        reg_in, patch_in = self._split_reg(tokens, num_reg)

        # 推断 H, W
        N = patch_in.shape[1]
        if patch_hw is None:
            side = int(math.sqrt(N))
            assert side * side == N, f"patch tokens ({N}) 不是正方形网格，需明确传入 patch_hw"
            patch_hw = (side, side)
        H, W = patch_hw

        # ========== S1（基尺度，含 reg） ==========
        # 在序列域先过一遍 Mamba（包含 reg）
        s1_tokens_out = self.stack_s1(tokens)       # (B, num_reg + H*W, C)
        reg_out, s1_patch_tokens = self._split_reg(s1_tokens_out, num_reg)
        s1_grid = self._tokens_to_grid(s1_patch_tokens, (H, W))   # (B, C, H, W)

        branches: List[torch.Tensor] = [s1_grid]

        # ========== S2（1/2 尺度，不含 reg） ==========
        if 2 in self.scales:
            s2_grid_in = F.avg_pool2d(s1_grid, kernel_size=2, stride=2)   # (B, C, H/2, W/2)
            s2_tokens = self._grid_to_tokens(s2_grid_in)                  # (B, (H/2)*(W/2), C)
            s2_tokens = self._run_small_stack(s2_tokens, scale=2)
            s2_grid = self._tokens_to_grid(s2_tokens, (H // 2, W // 2))
            # 上采样回 HxW
            s2_up = F.interpolate(s2_grid, size=(H, W), mode='bilinear', align_corners=False)
            branches.append(s2_up)

        # ========== S3（1/4 尺度，不含 reg） ==========
        if 4 in self.scales:
            # 基于 S2（若存在）再下采；否则从 S1 连续两次 avg_pool2d
            base_grid = branches[1] if (2 in self.scales) else s1_grid
            # 注意：branches[1] 已经是 HxW 上采样结果，不能再下采！因此重新从 s1_grid 下采两次更稳妥：
            s3_grid_in = F.avg_pool2d(s1_grid, kernel_size=4, stride=4)   # (B, C, H/4, W/4)
            s3_tokens = self._grid_to_tokens(s3_grid_in)
            s3_tokens = self._run_small_stack(s3_tokens, scale=4)
            s3_grid = self._tokens_to_grid(s3_tokens, (H // 4, W // 4))
            s3_up = F.interpolate(s3_grid, size=(H, W), mode='bilinear', align_corners=False)
            branches.append(s3_up)

        # ========== 融合 ==========
        if self.fuse == "sum":
            fused = torch.stack(branches, dim=0).sum(dim=0)   # (B, C, H, W)
        else:  # 'concat'
            fused = torch.cat(branches, dim=1)                # (B, C*k, H, W)
            fused = self.fuse_proj(fused)                     # (B, C, H, W)

        # 回到 tokens，并与 reg 拼接
        patch_out = self._grid_to_tokens(fused)               # (B, H*W, C)
        tokens_out = self._cat_reg(reg_out, patch_out)        # (B, num_reg + H*W, C)
        return tokens_out
