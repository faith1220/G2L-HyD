# # Dinomaly/models_mambaAD/lss_decoder.py
# import torch
# import torch.nn as nn
# from .mdssm_block import MDSSMBlock

# class DALRLayerBlock(nn.Module):
#     def __init__(self, dim, num_hss=3):
#         super().__init__()
#         self.hss = nn.Sequential(*[MDSSMBlock(dim) for _ in range(num_hss)])
#         self.local5 = nn.Sequential(
#             nn.Conv2d(dim, dim, 1),
#             nn.Conv2d(dim, dim, 5, padding=2, groups=dim),
#             nn.Conv2d(dim, dim, 1),
#         )
#         self.local7 = nn.Sequential(
#             nn.Conv2d(dim, dim, 1),
#             nn.Conv2d(dim, dim, 7, padding=3, groups=dim),
#             nn.Conv2d(dim, dim, 1),
#         )
#         self.fuse = nn.Conv2d(dim * 3, dim, 1)

#     def forward(self, x):
#         g  = self.hss(x)
#         l5 = self.local5(x)
#         l7 = self.local7(x)
#         out = self.fuse(torch.cat([g, l5, l7], dim=1)) + x
#         return out

# class DALRDecoder(nn.Module):
#     """
#     三尺度解码；深度配置可按 [3,4,6,3] 的前三个 stage 来：stage2 的 num_hss=2，其余=3
#     dims: 与 HFPN 输出的通道数对齐，例如 (256, 512, 1024)
#     """
#     def __init__(self, dims=(256, 512, 1024), out_dim=256):
#         super().__init__()
#         d1, d2, d3 = dims
#         self.stage1 = nn.Sequential(*[DALRLayerBlock(d1, num_hss=3) for _ in range(3)])
#         self.stage2 = nn.Sequential(*[DALRLayerBlock(d2, num_hss=2) for _ in range(4)])
#         self.stage3 = nn.Sequential(*[DALRLayerBlock(d3, num_hss=3) for _ in range(6)])
#         self.head1  = nn.Conv2d(d1, out_dim, 1)
#         self.head2  = nn.Conv2d(d2, out_dim, 1)
#         self.head3  = nn.Conv2d(d3, out_dim, 1)

#     def forward(self, feats):
#         """
#         feats: list/tuple of [f1,f2,f3], 每个 [B,C,H,W]
#         return: [o1,o2,o3]，用于多尺度 MSE / 余弦图
#         """
#         f1, f2, f3 = feats
#         r1 = self.stage1(f1); o1 = self.head1(r1)
#         r2 = self.stage2(f2); o2 = self.head2(r2)
#         r3 = self.stage3(f3); o3 = self.head3(r3)
#         return [o1, o2, o3]


# Dinomaly/models_mambaAD/lss_decoder.py
import torch
import torch.nn as nn
from .mdssm_block import MDSSMBlock, _DEFAULT_DIRS
from dalr.hilbert_scan import get_scan_directions

def _local_branch(dim, k=5, d=1):
    pad = ((k - 1) // 2) * d
    return nn.Sequential(
        nn.Conv2d(dim, dim, 1),
        nn.Conv2d(dim, dim, k, padding=pad, groups=dim, dilation=d),
        nn.Conv2d(dim, dim, 1),
    )

def _local_branch_asym(dim, k=7, d=1):
    # 1×k 与 k×1，增强细长结构（cable）
    pad = ((k - 1) // 2) * d
    return nn.Sequential(
        nn.Conv2d(dim, dim, 1),
        nn.Conv2d(dim, dim, (1, k), padding=(0, pad), groups=dim, dilation=(1, d)),
        nn.Conv2d(dim, dim, (k, 1), padding=(pad, 0), groups=dim, dilation=(d, 1)),
        nn.Conv2d(dim, dim, 1),
    )

class DALRLayerBlock(nn.Module):
    def __init__(self, dim, num_hss=3, ks=(5,7), dilations=(1,1),
                 add_local3=True, add_asym=True,
                 hss_learn_dir=True,
                 scan_method='hilbert', num_scan_dirs=8):
        super().__init__()

        hss_dirs = get_scan_directions(num_scan_dirs)
        
        # HSS：传入新参数（方向权重学习）
        self.hss = nn.Sequential(*[
            MDSSMBlock(dim, directions=hss_dirs, learn_dir=hss_learn_dir, use_conv_gate=True, scan_method=scan_method)
            for _ in range(num_hss)
        ])

        # 本地分支：可配置核与膨胀，额外加 3x3 & 各向异性分支
        branches = []
        if add_local3:
            branches.append(_local_branch(dim, k=3, d=1))
        for k, d in zip(ks, dilations):
            branches.append(_local_branch(dim, k=k, d=d))
        if add_asym:
            # 用最大的 k 做各向异性
            kmax = max(ks) if len(ks) > 0 else 7
            branches.append(_local_branch_asym(dim, k=kmax, d=1))
        self.locals = nn.ModuleList(branches)

        self.fuse = nn.Conv2d(dim * (1 + len(self.locals)), dim, 1)

    def forward(self, x):
        g = self.hss(x)
        outs = [g] + [b(x) for b in self.locals]
        out = self.fuse(torch.cat(outs, dim=1)) + x
        return out

class DALRDecoder(nn.Module):
    """
    三尺度解码；默认 stage2 的 num_hss=2，其余=3
    dims: 与 HFPN 输出的通道数对齐，例如 (256, 512, 1024)
    """
    def __init__(self, dims=(256, 512, 1024), out_dim=256,
                 # 每个 stage 的核与膨胀配置（可按需调整）
                 ks_cfg=((3,5), (5,7), (5,7)),
                 dil_cfg=((1,1), (1,2), (1,2)),
                 add_local3=(True, True, True),
                 add_asym=(True, True, True),
                 hss_learn_dir=True,
                 scan_method='hilbert', num_scan_dirs=8):
        super().__init__()
        d1, d2, d3 = dims

        self.stage1 = nn.Sequential(*[
            DALRLayerBlock(d1, num_hss=3, ks=ks_cfg[0], dilations=dil_cfg[0],
                     add_local3=add_local3[0], add_asym=add_asym[0],
                     hss_learn_dir=hss_learn_dir,
                     scan_method=scan_method, num_scan_dirs=num_scan_dirs)
            for _ in range(3)
        ])
        self.stage2 = nn.Sequential(*[
            DALRLayerBlock(d2, num_hss=2, ks=ks_cfg[1], dilations=dil_cfg[1],
                     add_local3=add_local3[1], add_asym=add_asym[1],
                     hss_learn_dir=hss_learn_dir,
                     scan_method=scan_method, num_scan_dirs=num_scan_dirs)
            for _ in range(4)
        ])
        self.stage3 = nn.Sequential(*[
            DALRLayerBlock(d3, num_hss=3, ks=ks_cfg[2], dilations=dil_cfg[2],
                     add_local3=add_local3[2], add_asym=add_asym[2],
                     hss_learn_dir=hss_learn_dir,
                     scan_method=scan_method, num_scan_dirs=num_scan_dirs)
            for _ in range(6)
        ])

        self.head1  = nn.Conv2d(d1, out_dim, 1)
        self.head2  = nn.Conv2d(d2, out_dim, 1)
        self.head3  = nn.Conv2d(d3, out_dim, 1)

    def forward(self, feats):
        """
        feats: list/tuple of [f1,f2,f3], 每个 [B,C,H,W]
        return: [o1,o2,o3]
        """
        f1, f2, f3 = feats
        r1 = self.stage1(f1); o1 = self.head1(r1)
        r2 = self.stage2(f2); o2 = self.head2(r2)
        r3 = self.stage3(f3); o3 = self.head3(r3)
        return [o1, o2, o3]
