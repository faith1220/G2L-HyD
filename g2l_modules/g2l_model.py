import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.batchnorm import _BatchNorm
from sklearn.cluster import KMeans
import math

# [ADDED] 引入我们实现的 LSS（内部含 HSS/Hilbert/多核 DWConv）
from dalr.dalr_decoder import DALRLayerBlock  # <-- 新增
from typing import Optional, Tuple


class G2LHyD(nn.Module):
    def __init__(
            self,
            encoder,
            bottleneck,
            decoder,
            target_layers=[2, 3, 4, 5, 6, 7, 8, 9],
            fuse_layer_encoder=[[0, 1, 2, 3, 4, 5, 6, 7]],
            fuse_layer_decoder=[[0, 1, 2, 3, 4, 5, 6, 7]],
            mask_neighbor_size=0,
            remove_class_token=False,
            encoder_require_grad_layer=[],
            fuse_fn=None,
    ) -> None:
        super(G2LHyD, self).__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.fuse_layer_decoder = fuse_layer_decoder
        self.remove_class_token = remove_class_token
        self.encoder_require_grad_layer = encoder_require_grad_layer

        if not hasattr(self.encoder, 'num_register_tokens'):
            self.encoder.num_register_tokens = 0
        self.mask_neighbor_size = mask_neighbor_size
        self.fuse_fn = fuse_fn

    def forward(self, x):
        x = self.encoder.prepare_tokens(x)
        en_list = []
        for i, blk in enumerate(self.encoder.blocks):
            if i <= self.target_layers[-1]:
                if i in self.encoder_require_grad_layer:
                    x = blk(x)
                else:
                    with torch.no_grad():
                        x = blk(x)
            else:
                continue
            if i in self.target_layers:
                en_list.append(x)
        side = int(math.sqrt(en_list[0].shape[1] - 1 - self.encoder.num_register_tokens))

        if self.remove_class_token:
            en_list = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en_list]

        x = self.fuse_feature(en_list)
        for i, blk in enumerate(self.bottleneck):
            x = blk(x)

        if self.mask_neighbor_size > 0:
            attn_mask = self.generate_mask(side, x.device)
        else:
            attn_mask = None

        de_list = []
        for i, blk in enumerate(self.decoder):
            x = blk(x, attn_mask=attn_mask)
            de_list.append(x)
        de_list = de_list[::-1]

        en = [self.fuse_feature([en_list[idx] for idx in idxs]) for idxs in self.fuse_layer_encoder]
        de = [self.fuse_feature([de_list[idx] for idx in idxs]) for idxs in self.fuse_layer_decoder]

        if not self.remove_class_token:  # class tokens have not been removed above
            en = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en]
            de = [d[:, 1 + self.encoder.num_register_tokens:, :] for d in de]

        en = [e.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for e in en]
        de = [d.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for d in de]
        return en, de

    def fuse_feature(self, feat_list):
        if self.fuse_fn is not None:
            return self.fuse_fn(feat_list)
        return torch.stack(feat_list, dim=1).mean(dim=1)

    def generate_mask(self, feature_size, device='cuda'):
        """
        Generate a square mask for the sequence. The masked positions are filled with float('-inf').
        Unmasked positions are filled with float(0.0).
        """
        h, w = feature_size, feature_size
        hm, wm = self.mask_neighbor_size, self.mask_neighbor_size
        mask = torch.ones(h, w, h, w, device=device)
        for idx_h1 in range(h):
            for idx_w1 in range(w):
                idx_h2_start = max(idx_h1 - hm // 2, 0)
                idx_h2_end = min(idx_h1 + hm // 2 + 1, h)
                idx_w2_start = max(idx_w1 - wm // 2, 0)
                idx_w2_end = min(idx_w1 + wm // 2 + 1, w)
                mask[
                idx_h1, idx_w1, idx_h2_start:idx_h2_end, idx_w2_start:idx_w2_end
                ] = 0
        mask = mask.view(h * w, h * w)
        if self.remove_class_token:
            return mask
        mask_all = torch.ones(h * w + 1 + self.encoder.num_register_tokens,
                              h * w + 1 + self.encoder.num_register_tokens, device=device)
        mask_all[1 + self.encoder.num_register_tokens:, 1 + self.encoder.num_register_tokens:] = mask
        return mask_all


class G2LHyDCat(nn.Module):
    def __init__(
            self,
            encoder,
            bottleneck,
            decoder,
            target_layers=[2, 3, 4, 5, 6, 7, 8, 9],
            fuse_layer_encoder=[1, 3, 5, 7],
            mask_neighbor_size=0,
            remove_class_token=False,
            encoder_require_grad_layer=[],
    ) -> None:
        super(G2LHyDCat, self).__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.remove_class_token = remove_class_token
        self.encoder_require_grad_layer = encoder_require_grad_layer

        if not hasattr(self.encoder, 'num_register_tokens'):
            self.encoder.num_register_tokens = 0
        self.mask_neighbor_size = mask_neighbor_size

    def forward(self, x):
        x = self.encoder.prepare_tokens(x)
        en_list = []
        for i, blk in enumerate(self.encoder.blocks):
            if i <= self.target_layers[-1]:
                if i in self.encoder_require_grad_layer:
                    x = blk(x)
                else:
                    with torch.no_grad():
                        x = blk(x)
            else:
                continue
            if i in self.target_layers:
                en_list.append(x)
        side = int(math.sqrt(en_list[0].shape[1] - 1 - self.encoder.num_register_tokens))

        if self.remove_class_token:
            en_list = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en_list]

        x = self.fuse_feature(en_list)
        for i, blk in enumerate(self.bottleneck):
            x = blk(x)

        for i, blk in enumerate(self.decoder):
            x = blk(x)

        en = [torch.cat([en_list[idx] for idx in self.fuse_layer_encoder], dim=2)]
        de = [x]

        if not self.remove_class_token:  # class tokens have not been removed above
            en = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en]
            de = [d[:, 1 + self.encoder.num_register_tokens:, :] for d in de]

        en = [e.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for e in en]
        de = [d.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for d in de]
        return en, de

    def fuse_feature(self, feat_list):
        return torch.stack(feat_list, dim=1).mean(dim=1)

class ViTAD(nn.Module):
    def __init__(
            self,
            encoder,
            bottleneck,
            decoder,
            target_layers=[2, 5, 8, 11],
            fuse_layer_encoder=[0, 1, 2],
            fuse_layer_decoder=[2, 5, 8],
            mask_neighbor_size=0,
            remove_class_token=False,
    ) -> None:
        super(ViTAD, self).__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder
        self.target_layers = target_layers
        self.fuse_layer_encoder = fuse_layer_encoder
        self.fuse_layer_decoder = fuse_layer_decoder
        self.remove_class_token = remove_class_token

        if not hasattr(self.encoder, 'num_register_tokens'):
            self.encoder.num_register_tokens = 0
        self.mask_neighbor_size = mask_neighbor_size

    def forward(self, x):
        x = self.encoder.prepare_tokens(x)
        en_list = []
        for i, blk in enumerate(self.encoder.blocks):
            if i <= self.target_layers[-1]:
                with torch.no_grad():
                    x = blk(x)
            else:
                continue
            if i in self.target_layers:
                en_list.append(x)
        side = int(math.sqrt(en_list[0].shape[1] - 1 - self.encoder.num_register_tokens))

        if self.remove_class_token:
            en_list = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en_list]
            x = x[:, 1 + self.encoder.num_register_tokens:, :]

        # x = torch.cat(en_list, dim=2)
        for i, blk in enumerate(self.bottleneck):
            x = blk(x)

        if self.mask_neighbor_size > 0:
            attn_mask = self.generate_mask(side, x.device)
        else:
            attn_mask = None

        de_list = []
        for i, blk in enumerate(self.decoder):
            x = blk(x, attn_mask=attn_mask)
            de_list.append(x)
        de_list = de_list[::-1]

        en = [en_list[idx] for idx in self.fuse_layer_encoder]
        de = [de_list[idx] for idx in self.fuse_layer_decoder]

        if not self.remove_class_token:  # class tokens have not been removed above
            en = [e[:, 1 + self.encoder.num_register_tokens:, :] for e in en]
            de = [d[:, 1 + self.encoder.num_register_tokens:, :] for d in de]

        en = [e.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for e in en]
        de = [d.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for d in de]
        return en, de


class G2LHyDv2(nn.Module):
    def __init__(
            self,
            encoder,
            bottleneck,
            decoder,
            target_layers=[2, 3, 4, 5, 6, 7]
    ) -> None:
        super(G2LHyDv2, self).__init__()
        self.encoder = encoder
        self.bottleneck = bottleneck
        self.decoder = decoder
        self.target_layers = target_layers
        if not hasattr(self.encoder, 'num_register_tokens'):
            self.encoder.num_register_tokens = 0

    def forward(self, x):
        x = self.encoder.prepare_tokens(x)
        en = []
        for i, blk in enumerate(self.encoder.blocks):
            if i <= self.target_layers[-1]:
                with torch.no_grad():
                    x = blk(x)
            else:
                continue
            if i in self.target_layers:
                en.append(x)

        x = self.fuse_feature(en)
        for i, blk in enumerate(self.bottleneck):
            x = blk(x)

        de = []
        for i, blk in enumerate(self.decoder):
            x = blk(x)
            de.append(x)

        side = int(math.sqrt(x.shape[1]))

        en = [e[:, self.encoder.num_register_tokens + 1:, :] for e in en]
        de = [d[:, self.encoder.num_register_tokens + 1:, :] for d in de]

        en = [e.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for e in en]
        de = [d.permute(0, 2, 1).reshape([x.shape[0], -1, side, side]).contiguous() for d in de]

        return en[::-1], de

    def fuse_feature(self, feat_list):
        return torch.stack(feat_list, dim=1).mean(dim=1)


class G2LHyDv3(nn.Module):
    def __init__(
            self,
            teacher,
            student,
            target_layers=[2, 3, 4, 5, 6, 7, 8, 9],
            fuse_dropout=0.,
    ) -> None:
        super(G2LHyDv3, self).__init__()
        self.teacher = teacher
        self.student = student
        if fuse_dropout > 0:
            self.fuse_dropout = nn.Dropout(fuse_dropout)
        else:
            self.fuse_dropout = nn.Identity()
        self.target_layers = target_layers
        if not hasattr(self.teacher, 'num_register_tokens'):
            self.teacher.num_register_tokens = 0

    def forward(self, x):
        with torch.no_grad():
            patch = self.teacher.prepare_tokens(x)
            x = patch
            en = []
            for i, blk in enumerate(self.teacher.blocks):
                if i <= self.target_layers[-1]:
                    x = blk(x)
                else:
                    continue
                if i in self.target_layers:
                    en.append(x)
            en = self.fuse_feature(en, fuse_dropout=False)

        x = patch
        de = []
        for i, blk in enumerate(self.student):
            x = blk(x)
            if i in self.target_layers:
                de.append(x)
        de = self.fuse_feature(de, fuse_dropout=False)

        en = en[:, 1 + self.teacher.num_register_tokens:, :]
        de = de[:, 1 + self.teacher.num_register_tokens:, :]
        side = int(math.sqrt(en.shape[1]))

        en = en.permute(0, 2, 1).reshape([x.shape[0], -1, side, side])
        de = de.permute(0, 2, 1).reshape([x.shape[0], -1, side, side])
        return [en.contiguous()], [de.contiguous()]

    def fuse_feature(self, feat_list, fuse_dropout=False):
        if fuse_dropout:
            feat = torch.stack(feat_list, dim=1)
            feat = self.fuse_dropout(feat).mean(dim=1)
            return feat
        else:
            return torch.stack(feat_list, dim=1).mean(dim=1)


class ReContrast(nn.Module):
    def __init__(
            self,
            encoder,
            encoder_freeze,
            bottleneck,
            decoder,
    ) -> None:
        super(ReContrast, self).__init__()
        self.encoder = encoder
        self.encoder.layer4 = None
        self.encoder.fc = None

        self.encoder_freeze = encoder_freeze
        self.encoder_freeze.layer4 = None
        self.encoder_freeze.fc = None

        self.bottleneck = bottleneck
        self.decoder = decoder

    def forward(self, x):
        en = self.encoder(x)
        with torch.no_grad():
            en_freeze = self.encoder_freeze(x)
        en_2 = [torch.cat([a, b], dim=0) for a, b in zip(en, en_freeze)]
        de = self.decoder(self.bottleneck(en_2))
        de = [a.chunk(dim=0, chunks=2) for a in de]
        de = [de[0][0], de[1][0], de[2][0], de[3][1], de[4][1], de[5][1]]
        return en_freeze + en, de

    def train(self, mode=True, encoder_bn_train=True):
        self.training = mode
        if mode is True:
            if encoder_bn_train:
                self.encoder.train(True)
            else:
                self.encoder.train(False)
            self.encoder_freeze.train(False)  # the frozen encoder is eval()
            self.bottleneck.train(True)
            self.decoder.train(True)
        else:
            self.encoder.train(False)
            self.encoder_freeze.train(False)
            self.bottleneck.train(False)
            self.decoder.train(False)
        return self


def update_moving_average(ma_model, current_model, momentum=0.99):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = update_average(old_weight, up_weight)

    for current_buffers, ma_buffers in zip(current_model.buffers(), ma_model.buffers()):
        old_buffer, up_buffer = ma_buffers.data, current_buffers.data
        ma_buffers.data = update_average(old_buffer, up_buffer, momentum)


def update_average(old, new, momentum=0.99):
    if old is None:
        return new
    return old * momentum + (1 - momentum) * new


def disable_running_stats(model):
    def _disable(module):
        if isinstance(module, _BatchNorm):
            module.backup_momentum = module.momentum
            module.momentum = 0

    model.apply(_disable)


def enable_running_stats(model):
    def _enable(module):
        if isinstance(module, _BatchNorm) and hasattr(module, "backup_momentum"):
            module.momentum = module.backup_momentum

    model.apply(_enable)
    
# ===== uad.py 新增类：DALRBlock =====
# class DALRBlock(nn.Module):
#     """
#     一个与 G2LHyD.decoder 兼容的“token 版 MambaAD 解码块”：
#     - 输入:  x: [B, N_tokens, C]（含 cls 与 register tokens）
#     - 处理:  仅对 patch tokens 做 LSS（HSS + DWConv5/7），cls/reg 原样旁路
#     - 输出:  x': [B, N_tokens, C]
#     这样 G2LHyD 的 de_list、fuse 逻辑与 utils 的评测/损失保持不变。
#     """
#     def __init__(
#         self,
#         embed_dim: int,
#         num_register_tokens: int = 0,
#         num_hss: int = 3,                 # 论文多数 stage 用 Mi=3；可在构造时改为 2 做 ablation
#     ):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.num_register_tokens = num_register_tokens
#         self.lss = DALRLayerBlock(dim=embed_dim, num_hss=num_hss)  # 我们的 LSS（含 HSS + 多核 DWConv）

#     def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
#         """
#         x: [B, N_tokens, C]，其中 N_tokens = 1(cls) + R(register) + H*W(patch)
#         """
#         B, N, C = x.shape
#         assert C == self.embed_dim, f"embed_dim mismatch: {C} vs {self.embed_dim}"

#         # 1) 拆出 cls & reg tokens（旁路）
#         head = x[:, : 1 + self.num_register_tokens, :]                    # [B, 1+R, C]
#         patch = x[:, 1 + self.num_register_tokens :, :]                   # [B, H*W, C]

#         # 2) 推断 patch 网格边长
#         L = patch.shape[1]
#         side = int(math.sqrt(L))
#         assert side * side == L, f"patch tokens ({L}) 不是正方形网格，请检查输入"

#         # 3) token -> map
#         feat = patch.transpose(1, 2).contiguous().view(B, C, side, side)  # [B,C,H,W]

#         # 4) 过 LSS（= HSS 串行 + DWConv5/7 并行 + 1x1 fuse + 残差）
#         feat = self.lss(feat)                                             # [B,C,H,W]

#         # 5) map -> token，接回 cls/reg
#         patch_out = feat.flatten(2).transpose(1, 2).contiguous()          # [B,H*W,C]
#         x_out = torch.cat([head, patch_out], dim=1)                       # [B,N_tokens,C]
#         return x_out


# 11月6号
# class DALRBlock(nn.Module):
#     """
#     与 G2LHyD.decoder 兼容的“token 版 MambaAD 解码块”：
#     - 输入:  x: [B, N_tokens, C]（含 cls 与 register tokens）
#     - 处理:  仅对 patch tokens 做 LSS（HSS + DWConv 多核 + 方向可学习），cls/reg 旁路
#     - 输出:  x': [B, N_tokens, C]
#     """
#     def __init__(
#         self,
#         embed_dim: int,
#         num_register_tokens: int = 0,
#         num_hss: int = 3,
#         lss_kwargs: dict | None = None,   # 可传各向异性核/膨胀/learn_dir 等
#     ):
#         super().__init__()
#         self.embed_dim = embed_dim
#         self.num_register_tokens = num_register_tokens
#         self.lss = DALRLayerBlock(
#             dim=embed_dim,
#             num_hss=num_hss,
#             **(lss_kwargs or {})
#         )

#     def forward(self, x: torch.Tensor, attn_mask: torch.Tensor | None = None) -> torch.Tensor:
#         """
#         x: [B, N_tokens, C]，N_tokens = 1(cls) + R(register) + H*W(patch)
#         """
#         B, N, C = x.shape
#         assert C == self.embed_dim, f"embed_dim mismatch: {C} vs {self.embed_dim}"

#         # 1) 拆出 cls & reg（旁路）
#         head  = x[:, : 1 + self.num_register_tokens, :]         # [B, 1+R, C]
#         patch = x[:, 1 + self.num_register_tokens :, :]         # [B, H*W, C]

#         # 2) 推断 patch 网格
#         L = patch.shape[1]
#         side = int(math.sqrt(L))
#         assert side * side == L, (
#             f"patch tokens={L} 不是正方形网格；请检查输入尺寸/patch_size，或为该路径提供(H,W)。"
#         )

#         # 3) token -> map
#         feat = patch.transpose(1, 2).contiguous().view(B, C, side, side)  # [B,C,H,W]

#         # 4) 过 LSS（HSS 串行 + 多核 DWConv + 1x1 fuse + 残差）
#         feat = self.lss(feat)                                             # [B,C,H,W]

#         # 5) map -> token，接回 cls/reg
#         patch_out = feat.flatten(2).transpose(1, 2).contiguous()          # [B,H*W,C]
#         x_out = torch.cat([head, patch_out], dim=1)                       # [B,N_tokens,C]
#         return x_out


# 新增参数卷积核
class DALRBlock(nn.Module):
    """
    与 G2LHyD.decoder 兼容的“token 版 MambaAD 解码块”：
    - 输入:  x: [B, N_tokens, C]（含 cls 与 register tokens）
    - 处理:  仅对 patch tokens 做 LSS（HSS + DWConv 多核 + 方向可学习），cls/reg 旁路
    - 输出:  x': [B, N_tokens, C]

    可配置项（经由 lss_kwargs 传入，也可运行时 reconfigure_lss 改动）：
        ks:            (k1, k2) 两条本地卷积分支核，如 (5,7) 或 (3,3)
        dilations:     (d1, d2) 两条分支的膨胀系数，如 (1,2)
        add_local3:    bool，是否额外加入 3×3 分支
        add_asym:      bool，是否加入 1×k / k×1 的非对称卷积
        hss_learn_dir: bool，HSS 扫描方向是否可学习
    """

    def __init__(
        self,
        embed_dim: int,
        num_register_tokens: int = 0,
        num_hss: int = 3,
        scan_method: str = 'hilbert',
        num_scan_dirs: int = 8,
        lss_kwargs: Optional[dict] = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_register_tokens = num_register_tokens

        # —— 保存/规范化 LSS 配置（给热更新也能用到）
        if lss_kwargs is None:
            lss_kwargs = dict(ks=(5, 7), dilations=(1, 2),
                              add_local3=True, add_asym=True, hss_learn_dir=True)
        self.lss_cfg = self._normalize_lss_cfg(lss_kwargs)
        self.num_hss = int(num_hss)
        self.scan_method = scan_method
        self.num_scan_dirs = num_scan_dirs

        # —— 构建 LSS 模块
        self.lss = DALRLayerBlock(
            dim=embed_dim,
            num_hss=self.num_hss,
            scan_method=self.scan_method,
            num_scan_dirs=self.num_scan_dirs,
            **self.lss_cfg
        )

    # ========== 热更新：训练/评测期可随时改核 ==========
    @torch.no_grad()
    def reconfigure_lss(self,
                        ks: Optional[Tuple[int, int]] = None,
                        dilations: Optional[Tuple[int, int]] = None,
                        add_local3: Optional[bool] = None,
                        add_asym: Optional[bool] = None,
                        hss_learn_dir: Optional[bool] = None,
                        num_hss: Optional[int] = None):
        cfg = dict(self.lss_cfg)  # 当前配置
        if ks is not None:           cfg["ks"] = self._to_pair(ks)
        if dilations is not None:    cfg["dilations"] = self._to_pair(dilations)
        if add_local3 is not None:   cfg["add_local3"] = bool(add_local3)
        if add_asym is not None:     cfg["add_asym"] = bool(add_asym)
        if hss_learn_dir is not None:cfg["hss_learn_dir"] = bool(hss_learn_dir)
        if num_hss is not None:      self.num_hss = int(num_hss)

        # 仅当有变化时重建
        if cfg != self.lss_cfg:
            self.lss_cfg = cfg
            device, dtype = next(self.parameters()).device, next(self.parameters()).dtype
            self.lss = DALRLayerBlock(dim=self.embed_dim, num_hss=self.num_hss, **self.lss_cfg).to(device=device, dtype=dtype)

    # ========== 前向 ==========
    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None,
                hw: Optional[Tuple[int, int]] = None, H: Optional[int] = None, W: Optional[int] = None) -> torch.Tensor:
        """
        x:  [B, N_tokens, C]，N_tokens = 1(cls) + R(register) + H*W(patch)
        hw/HW：可显式传入 patch 网格尺寸（支持矩形）。若均未提供，将自动搜索 L 的因子对。
        """
        B, N, C = x.shape
        assert C == self.embed_dim, f"embed_dim mismatch: {C} vs {self.embed_dim}"

        # 1) 拆出 cls & reg（旁路）
        head_len = 1 + self.num_register_tokens
        head  = x[:, :head_len, :]                       # [B, 1+R, C]
        patch = x[:, head_len:, :]                       # [B, H*W, C]
        L = patch.shape[1]

        # 2) 推断 patch 网格
        if hw is not None:
            H_, W_ = int(hw[0]), int(hw[1])
            assert H_ * W_ == L, f"provided hw=({H_},{W_}) but tokens={L}"
        elif (H is not None) and (W is not None):
            H_, W_ = int(H), int(W)
            assert H_ * W_ == L, f"provided H,W=({H_},{W_}) but tokens={L}"
        else:
            # 自动寻找最接近平方的因子对（避免非平方时报错）
            H_, W_ = self._closest_factors(L)

        # 3) token -> map
        feat = patch.transpose(1, 2).contiguous().view(B, C, H_, W_)  # [B,C,H,W]

        # 4) 过 LSS（HSS 串行 + 多核 DWConv + 1x1 fuse + 残差）
        feat = self.lss(feat)                                        # [B,C,H,W]

        # 5) map -> token，接回 cls/reg
        patch_out = feat.flatten(2).transpose(1, 2).contiguous()     # [B,H*W,C]
        x_out = torch.cat([head, patch_out], dim=1)                  # [B,N_tokens,C]
        return x_out

    # ========== 工具 ==========
    @staticmethod
    def _to_pair(v):
        if isinstance(v, (list, tuple)) and len(v) == 2:
            return (int(v[0]), int(v[1]))
        v = int(v)
        return (v, v)

    @staticmethod
    def _normalize_lss_cfg(cfg_in: dict) -> dict:
        cfg = dict(cfg_in)
        cfg["ks"] = DALRBlock._to_pair(cfg.get("ks", (5, 7)))
        cfg["dilations"] = DALRBlock._to_pair(cfg.get("dilations", (1, 2)))
        cfg["add_local3"] = bool(cfg.get("add_local3", True))
        cfg["add_asym"] = bool(cfg.get("add_asym", True))
        cfg["hss_learn_dir"] = bool(cfg.get("hss_learn_dir", True))
        return cfg

    @staticmethod
    def _closest_factors(n: int) -> Tuple[int, int]:
        """为 n 找到一对最接近 sqrt(n) 的整数因子 (H,W)。"""
        s = int(math.sqrt(n))
        if s * s == n:
            return s, s
        # 从 sqrt(n) 向下找因子
        for h in range(s, 0, -1):
            if n % h == 0:
                return h, n // h
        # 理论上不会走到这，兜底成 1×n
        return 1, n
