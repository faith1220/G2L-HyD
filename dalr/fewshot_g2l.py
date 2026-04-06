# # Dinomaly/models_mambaAD/fewshot_mambaad.py
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from .utils.multiscale import HFPN
# from .dalr_decoder import DALRDecoder

# def _global_vecs(feats):
#     # feats: list of [B,C,H,W] -> [B, sumC]
#     return torch.cat([F.adaptive_avg_pool2d(f,1).flatten(1) for f in feats], dim=1)

# class FewShotG2L(nn.Module):
#     def __init__(self, encoder, feat_dims, out_dim=256, sim_weight=0.5, rec_weight=0.5):
#         """
#         encoder: 需提供多层特征输出（你可以包装成一个带 extract_features(x)->list[3 tensors] 的对象）
#         feat_dims: encoder 多层通道数，例如 [384, 768, 1024] (DINOv2-S/B 需按实际填)
#         """
#         super().__init__()
#         self.encoder = encoder.eval()
#         for p in self.encoder.parameters(): p.requires_grad = False

#         self.hfpn = HFPN(in_dims=feat_dims, out_dims=(256, 512, 1024))
#         self.decoder = DALRDecoder(dims=(256, 512, 1024), out_dim=out_dim)

#         self.sim_w, self.rec_w = sim_weight, rec_weight
#         self.register_buffer('mem_feats', None)  # [K, D] 的全局向量记忆

#     @torch.no_grad()
#     def _extract_multiscale(self, x):
#         """
#         适配器：尽量兼容多种 encoder 写法
#         期望输出 raw_feats: [f1,f2,f3]，每个 [B,C,H,W]
#         """
#         if hasattr(self.encoder, 'extract_features'):
#             raw_feats = self.encoder.extract_features(x)
#         elif hasattr(self.encoder, 'forward_features'):
#             raw_feats = self.encoder.forward_features(x)
#         else:
#             raw = self.encoder(x)
#             if isinstance(raw, (list, tuple)):
#                 raw_feats = raw
#             else:
#                 raise RuntimeError("Encoder must provide extract_features/forward_features or return multi-scale list.")
#         assert isinstance(raw_feats, (list, tuple)) and len(raw_feats) >= 3, \
#             "Need at least 3 scales from encoder"
#         # 取后三个尺度（通常分辨率从高到低）
#         feats3 = list(raw_feats)[-3:]
#         feats3 = self.hfpn(feats3)  # 通道对齐
#         return feats3

#     @torch.no_grad()
#     def build_memory(self, imgs):
#         feats = self._extract_multiscale(imgs)      # list [B,C,H,W]
#         g = _global_vecs(feats)                      # [B, D]
#         self.mem_feats = g if self.mem_feats is None else torch.cat([self.mem_feats, g], dim=0)

#     def forward(self, x):
#         feats = self._extract_multiscale(x)
#         dec_feats = self.decoder(feats)
#         return feats, dec_feats

#     def anomaly_score(self, feats, dec_feats):
#         # 多尺度重建误差（像素级）
#         rec_map = 0
#         for f, d in zip(feats, dec_feats):
#             f_ = F.interpolate(f, size=d.shape[-2:], mode='bilinear', align_corners=False)
#             rec_map = rec_map + (f_ - d).pow(2).mean(dim=1, keepdim=True)  # [B,1,H,W]
#         # 记忆相似度（图像级）
#         g = _global_vecs(feats)                   # [B, D]
#         if self.mem_feats is None:
#             sim_map = torch.zeros_like(rec_map)
#         else:
#             # 距离 = 1 - max cosine，相当于异常得分
#             cs = F.cosine_similarity(g.unsqueeze(1), self.mem_feats.unsqueeze(0), dim=-1)  # [B,K]
#             sim = 1 - cs.max(dim=1, keepdim=True)[0]  # [B,1]
#             sim_map = sim.view(-1,1,1,1).expand_as(rec_map)

#         score = self.sim_w * sim_map + self.rec_w * rec_map
#         return score  # [B,1,H,W]

# Dinomaly/models_mambaAD/fewshot_mambaad.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .utils.multiscale import HFPN
from .dalr_decoder import DALRDecoder

def _global_vecs(feats):
    """
    feats: list of [B,C,H,W] -> [B, sumC]
    全局池化后拼接，用于 few-shot 记忆库相似度
    """
    return torch.cat([F.adaptive_avg_pool2d(f, 1).flatten(1) for f in feats], dim=1)

class FewShotG2L(nn.Module):
    """
    Few-shot 推断期模型：
      - 冻结 encoder，取多尺度特征 -> HFPN 通道对齐 -> LSS 解码器
      - 重建误差 (像素级) + 记忆库相似度 (图像级) 融合为异常图
    关键修复：
      - 为每个尺度加 1×1 投影到 out_dim，确保与解码头通道一致再做 MSE
    """

    def __init__(
        self,
        encoder,
        feat_dims,                   # 例如 [384, 768, 1024]（按你的 encoder 实际输出填写）
        out_dim: int = 256,
        sim_weight: float = 0.5,     # 图像级相似度权重
        rec_weight: float = 0.5,     # 像素级重建权重
        rec_scale_weights=(1.0, 1.0, 1.0),  # 三个尺度的重建权重
        lss_cfg: Optional[dict] = None, # 透传给 DALRDecoder 的可选配置（核大小/膨胀/各向异性/HSS方向学习等）
        hfpn_out_dims=(256, 512, 1024),     # HFPN 对齐后的通道
        freeze_encoder: bool = True,
    ):
        super().__init__()

        # === 1) 冻结 encoder（评测期仅提特征）
        self.encoder = encoder
        if freeze_encoder:
            self.encoder.eval()
            for p in self.encoder.parameters():
                p.requires_grad = False

        # === 2) 多尺度通道对齐
        self.hfpn = HFPN(in_dims=feat_dims, out_dims=hfpn_out_dims)
        d1, d2, d3 = hfpn_out_dims

        # === 3) LSS 解码器（已在 lss_decoder.py 内支持：方向可学习 + 3x3/各向异性核）
        if lss_cfg is None:
            # 通用稳健版（建议做新基线；你也可以在外部按类切换为“专治 cable / screw/capsule”的那两套）
            lss_cfg = dict(
                ks_cfg=((3, 5), (5, 7), (5, 7)),
                dil_cfg=((1, 1), (1, 2), (1, 2)),
                add_local3=(True, True, True),
                add_asym=(True, True, True),
                hss_learn_dir=True,
            )
        self.decoder = DALRDecoder(dims=(d1, d2, d3), out_dim=out_dim, **lss_cfg)

        # === 4) 通道对齐到 out_dim 再做重建误差（关键修复点）
        self.enc_proj = nn.ModuleList([
            nn.Conv2d(d1, out_dim, 1, bias=True),
            nn.Conv2d(d2, out_dim, 1, bias=True),
            nn.Conv2d(d3, out_dim, 1, bias=True),
        ])

        # === 5) 融合权重与 few-shot 记忆库
        self.sim_w, self.rec_w = float(sim_weight), float(rec_weight)
        self.register_buffer('rec_scale_w', torch.tensor(rec_scale_weights, dtype=torch.float32).view(3, 1, 1, 1))
        self.register_buffer('mem_feats', None)  # [K, D] 的全局向量记忆（自适应构建）

    # --------------------- 内部：提多尺度特征 --------------------- #
    @torch.no_grad()
    def _extract_multiscale(self, x):
        """
        适配器：兼容多种 encoder 写法
        期望输出 raw_feats: list/tuple of [B,C,H,W], 取其中后三个尺度
        """
        if hasattr(self.encoder, 'extract_features'):
            raw_feats = self.encoder.extract_features(x)
        elif hasattr(self.encoder, 'forward_features'):
            raw_feats = self.encoder.forward_features(x)
        else:
            raw = self.encoder(x)
            if isinstance(raw, (list, tuple)):
                raw_feats = raw
            else:
                raise RuntimeError("Encoder must provide extract_features/forward_features or return multi-scale list.")
        assert isinstance(raw_feats, (list, tuple)) and len(raw_feats) >= 3, \
            "Need at least 3 scales from encoder"
        feats3 = list(raw_feats)[-3:]
        feats3 = self.hfpn(feats3)  # 通道对齐到 (d1,d2,d3)
        return feats3

    # --------------------- Few-shot 记忆构建 --------------------- #
    @torch.no_grad()
    def build_memory(self, imgs):
        """
        imgs: [B,3,H,W] 一批支持图（正常样本）——构建/追加到记忆库
        记忆向量为多尺度全局池化后的拼接（与 anomaly_score 一致）
        """
        feats = self._extract_multiscale(imgs)             # list [B,C,H,W]
        g = _global_vecs(feats).detach()                   # [B, D]
        if self.mem_feats is None:
            self.mem_feats = g
        else:
            self.mem_feats = torch.cat([self.mem_feats, g], dim=0)

    # --------------------- 前向：回传多尺度 & 解码器输出 --------------------- #
    def forward(self, x):
        feats = self._extract_multiscale(x)      # 上游特征（对齐后通道）
        dec_feats = self.decoder(feats)          # 解码器输出 [o1,o2,o3]，每个 [B,out_dim,H,W]
        return feats, dec_feats

    # --------------------- 异常图与分数 --------------------- #
    def anomaly_score(self, feats, dec_feats, eps: float = 1e-6):
        """
        输入：
          feats:  _extract_multiscale(x) 的输出（[B,d*,H*,W*]）
          dec_feats: decoder(feats) 的输出（[B,out_dim,H,W]）
        输出：
          score: [B,1,H,W] 的异常热图（像素级），融合 rec + sim
        """
        assert len(feats) == len(dec_feats) == 3, "Expect 3-scale features"

        # ===== 1) 多尺度重建误差（先把 encoder 特征投影到 out_dim） =====
        rec_map = 0.0
        for i, (f, d) in enumerate(zip(feats, dec_feats)):
            # 投影到 out_dim
            f = self.enc_proj[i](f)  # [B,out_dim,h,w]
            # 空间对齐到解码输出分辨率
            f = F.interpolate(f, size=d.shape[-2:], mode='bilinear', align_corners=False)
            # MSE（你也可换 Charbonnier: torch.sqrt((f-d).pow(2) + eps) 再均值）
            mse = (f - d).pow(2).mean(dim=1, keepdim=True)  # [B,1,H,W]
            rec_map = rec_map + self.rec_scale_w[i] * mse

        # ===== 2) 图像级相似度（few-shot 记忆库） =====
        g = _global_vecs(feats)  # [B, D]
        if self.mem_feats is None or self.mem_feats.numel() == 0:
            sim_map = torch.zeros_like(rec_map)
        else:
            # 1 - 最大余弦相似度 作为“距离”/异常分数
            # F.cosine_similarity 本身做 L2 归一化，无需手动 normalize
            cs = F.cosine_similarity(g.unsqueeze(1), self.mem_feats.unsqueeze(0), dim=-1)  # [B,K]
            sim = 1.0 - cs.max(dim=1, keepdim=True)[0]  # [B,1]
            sim_map = sim.view(-1, 1, 1, 1).expand_as(rec_map)  # 拉成像素图以便融合

        # ===== 3) 融合 =====
        score = self.sim_w * sim_map + self.rec_w * rec_map
        return score  # [B,1,H,W]

