import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import warnings
warnings.filterwarnings("ignore")

import cv2
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import numpy as np
import random
import argparse
import logging

from functools import partial
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets import ImageFolder
from tabulate import tabulate

# ====== 项目内模块 ======
from dataset import get_data_transforms, get_strong_transforms
from dataset import MVTecDataset
from g2l_modules.g2l_model import G2LHyD, G2LHyDv2, DALRBlock   # ★ 引入 Mamba 解码块
from g2l_modules import encoder
from dinov1.utils import trunc_normal_
from g2l_modules.gfsr_blocks import (
    GFSRBlock, bMlp, Attention, LinearAttention, LinearAttention,DropoutSchedule,LinearGMLPTokenBlock,
    ConvBlock, FeatureJitter, CGB,SAB,
    NoisyGEGLUBottleneck, MaskTokenBottleneck, HFPNBottleneck, HVQBottle, RSCReconstructionHead
)
from dalr.fewshot_g2l import FewShotG2L   # ← 新增

from utils import (
    evaluation_batch, global_cosine, regional_cosine_hm_percent,
    global_cosine_hm_percent, WarmCosineScheduler,
    compute_image_score_from_heatmap, cal_anomaly_maps
)
from visualization import visualizer

from optimizers import StableAdamW
import copy
import itertools
import functools


# ------------------------- 基础工具 -------------------------
def get_logger(name, save_path=None, level='INFO'):
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level))
    log_format = logging.Formatter('%(message)s')
    streamHandler = logging.StreamHandler()
    streamHandler.setFormatter(log_format)
    if logger.handlers:
        logger.handlers.clear()
    logger.addHandler(streamHandler)
    if save_path is not None:
        os.makedirs(save_path, exist_ok=True)
        fileHandler = logging.FileHandler(os.path.join(save_path, 'log.txt'))
        fileHandler.setFormatter(log_format)
        logger.addHandler(fileHandler)
    return logger

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _safe_nanmean(values):
    arr = np.asarray(values, dtype=np.float64)
    return float(np.nanmean(arr)) if arr.size > 0 and not np.all(np.isnan(arr)) else float("nan")


def _log_metric_table(metric_rows, print_fn=print):
    headers = ["Class", "Image AUROC", "Image F1max", "Pixel AUROC", "Pixel AUPRO", "Pixel F1max"]
    table_rows = [
        [
            row["class"],
            row["i_auroc"],
            row["i_f1"],
            row["p_auroc"],
            row["p_aupro"],
            row["p_f1"],
        ]
        for row in metric_rows
    ]
    if metric_rows:
        table_rows.append([
            "Mean",
            _safe_nanmean([r["i_auroc"] for r in metric_rows]),
            _safe_nanmean([r["i_f1"] for r in metric_rows]),
            _safe_nanmean([r["p_auroc"] for r in metric_rows]),
            _safe_nanmean([r["p_aupro"] for r in metric_rows]),
            _safe_nanmean([r["p_f1"] for r in metric_rows]),
        ])
    print_fn("\n" + tabulate(table_rows, headers=headers, tablefmt="github", floatfmt=".4f"))


def _strip_module_prefix(state_dict):
    return {
        (k[7:] if k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }


def _extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        if "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
            return ckpt_obj["model"]
        if "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
            return ckpt_obj["state_dict"]
    return ckpt_obj


def _load_state_dict_with_fallback(model, state_dict, print_fn=print):
    """
    strict=False still fails on shape mismatch. This loader keeps only keys with
    matching tensor shapes so inference can continue when a few arch args differ.
    """
    state_dict = _strip_module_prefix(state_dict)
    model_state = model.state_dict()

    matched = {}
    mismatched = []
    unexpected = []
    for k, v in state_dict.items():
        if k not in model_state:
            unexpected.append(k)
            continue
        if model_state[k].shape != v.shape:
            mismatched.append((k, tuple(v.shape), tuple(model_state[k].shape)))
            continue
        matched[k] = v

    missing, unexpected_load = model.load_state_dict(matched, strict=False)
    unexpected.extend(list(unexpected_load))

    if mismatched:
        print_fn(
            f"[ckpt] shape-mismatch fallback enabled: kept={len(matched)}, "
            f"mismatched={len(mismatched)}, missing={len(missing)}, unexpected={len(unexpected)}"
        )
        for k, src_shape, dst_shape in mismatched[:12]:
            print_fn(f"[ckpt][mismatch] {k}: ckpt{src_shape} -> model{dst_shape}")
        if len(mismatched) > 12:
            print_fn(f"[ckpt] ... and {len(mismatched) - 12} more mismatched keys.")

    return missing, unexpected, mismatched


# --- 本地 token 拆并（给原型构建用） ---
def _split_tokens(x, num_regs=4, has_cls=True):
    """
    x: [B, N, C] -> (cls [B,1,C] or None, regs [B,R,C] or None, patches [B,HW,C])
    """
    if has_cls:
        cls, rest = x[:, :1], x[:, 1:]
    else:
        cls, rest = None, x
    regs = rest[:, :num_regs] if num_regs > 0 else None
    patches = rest[:, num_regs:] if num_regs > 0 else rest
    return cls, regs, patches


class ResidualScale(nn.Module):
    """
    y = x + gamma * (f(x) - x)
    用一个可学习的 gamma（默认0.2）把子模块的改变量“压一压”，防止一上来破坏分布。
    """
    def __init__(self, module: nn.Module, init_scale: float = 0.2):
        super().__init__()
        self.module = module
        self.gamma = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))

    def forward(self, x, **kwargs):
        y = self.module(x, **kwargs)
        return x + self.gamma * (y - x)

class PostNorm(nn.Module):
    """
    子模块输出后做一个 LayerNorm（eps更小），稳住数值范围。
    """
    def __init__(self, module: nn.Module, dim: int, eps: float = 1e-6):
        super().__init__()
        self.module = module
        self.ln = nn.LayerNorm(dim, eps=eps)

    def forward(self, x, **kwargs):
        y = self.module(x, **kwargs)
        return self.ln(y)


class LearnableFusionWeights(nn.Module):
    def __init__(self, num_groups: int, temperature: float = 1.0, init_weights=None):
        super().__init__()
        if init_weights is not None:
            w = torch.tensor(init_weights, dtype=torch.float32)
            w = w / (w.sum() + 1e-6)
            logits = torch.log(w + 1e-6)
        else:
            logits = torch.zeros(num_groups, dtype=torch.float32)
        self.logits = nn.Parameter(logits)
        self.temperature = float(temperature)

    def forward(self):
        temp = self.temperature if self.temperature > 1e-6 else 1e-6
        return torch.softmax(self.logits / temp, dim=0)


class SpatialFusionGate(nn.Module):
    def __init__(self, num_groups: int, hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(num_groups, hidden, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, num_groups, kernel_size=1),
        )

    def forward(self, x):
        return self.net(x)


def visualize_mvtec_results(
    model,
    test_data_list,
    item_list,
    device,
    workers: int = 0,
    save_root: str = "results",
    max_images: int = 0,
    vis_alpha: float = 0.5,
    vis_sigma: float = 4.0,
    fusion_mode: str = "mean",
    fusion_confidence: str = "topk_mean",
    fusion_topk: float = 0.02,
    fusion_temp: float = 0.07,
    fusion_weights=None,
    fusion_gate=None,
    print_fn=print,
):
    """
    Export per-sample original/heatmap/overlay to:
    <save_root>/imgs/<class>/<defect>/{original,heatmap,overlay}/filename
    """
    model.eval()
    if fusion_gate is not None:
        fusion_gate.eval()

    with torch.no_grad():
        for item, dataset in zip(item_list, test_data_list):
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=1, shuffle=False, num_workers=workers
            )

            for idx, (img, _, _, img_path) in enumerate(loader):
                img = img.to(device)
                anomaly_map, _ = cal_anomaly_maps(
                    *model(img),
                    img.shape[-1],
                    weights=fusion_weights,
                    fusion_mode=fusion_mode,
                    fusion_confidence=fusion_confidence,
                    fusion_topk=fusion_topk,
                    fusion_temp=fusion_temp,
                    fusion_gate=fusion_gate,
                )

                target_h, target_w = img.shape[-2], img.shape[-1]
                if anomaly_map.shape[-2:] != (target_h, target_w):
                    anomaly_map = F.interpolate(
                        anomaly_map, size=(target_h, target_w), mode="bilinear", align_corners=False
                    )

                amap_np = anomaly_map.squeeze(1).cpu().numpy().astype(np.float32)
                if vis_sigma > 0:
                    for b in range(amap_np.shape[0]):
                        amap_np[b] = cv2.GaussianBlur(
                            amap_np[b], ksize=(0, 0), sigmaX=vis_sigma, sigmaY=vis_sigma
                        )

                visualizer(
                    img_paths=img_path,
                    anomaly_map=amap_np,
                    img_size=(target_h, target_w),
                    save_path=save_root,
                    cls_name=item,
                    alpha=vis_alpha,
                )

                if max_images and (idx + 1) >= max_images:
                    break

            if print_fn is not None:
                print_fn(f"[visualize] {item}: exported")


def evaluate_mvtec_metrics(
    model,
    item_list,
    test_data_list,
    train_data_list,
    device,
    batch_size,
    workers,
    args,
    fusion_weights=None,
    fusion_gate=None,
    print_fn=print,
):
    # Step-1：选择图像级聚合器
    agg_fn = None
    if args.hotspot_gem:
        agg_fn = partial(
            compute_image_score_from_heatmap, k_ratio=args.topk_ratio, p=args.gem_p, fg_q=args.fg_q
        )
    elif args.i_agg == "max_topk":
        from utils import image_score_max_topk

        agg_fn = partial(image_score_max_topk, alpha=args.i_alpha, top_percent=args.i_top_percent)

    model.eval()
    if fusion_gate is not None:
        fusion_gate.eval()

    metric_rows = []
    auroc_sp_list, ap_sp_list, f1_sp_list = [], [], []
    auroc_px_list, ap_px_list, f1_px_list, aupro_px_list = [], [], [], []

    for item, test_data, calib_data in zip(item_list, test_data_list, train_data_list):
        test_dataloader = torch.utils.data.DataLoader(
            test_data, batch_size=batch_size, shuffle=False, num_workers=workers
        )

        results = evaluation_batch(
            model,
            test_dataloader,
            device,
            max_ratio=0.01,
            resize_mask=256,
            aggregator=agg_fn,
            flip_tta=args.flip_tta,
            z_norm=args.z_norm,
            z_calib_dataset=calib_data,
            postproc=args.postproc,
            topk_ratio=args.topk_ratio,
            gem_p=args.gem_p,
            fg_q=args.fg_q,
            post_k=args.post_k,
            post_iters=args.post_iters,
            rot_tta=args.rot_tta,
            ms_tta=args.ms_tta,
            group_weights=fusion_weights,
            fusion_mode=args.fusion_mode,
            fusion_confidence=args.fusion_confidence,
            fusion_topk=args.fusion_topk,
            fusion_temp=args.fusion_temp,
            fusion_gate=fusion_gate,
        )

        auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results
        auroc_sp_list.append(auroc_sp)
        ap_sp_list.append(ap_sp)
        f1_sp_list.append(f1_sp)
        auroc_px_list.append(auroc_px)
        ap_px_list.append(ap_px)
        f1_px_list.append(f1_px)
        aupro_px_list.append(aupro_px)

        metric_rows.append(
            {
                "class": item,
                "i_auroc": auroc_sp,
                "i_f1": f1_sp,
                "p_auroc": auroc_px,
                "p_aupro": aupro_px,
                "p_f1": f1_px,
            }
        )
        print_fn(
            "{}: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, "
            "P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}".format(
                item, auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px
            )
        )

    mean_metrics = (
        np.mean(auroc_sp_list),
        np.mean(ap_sp_list),
        np.mean(f1_sp_list),
        np.mean(auroc_px_list),
        np.mean(ap_px_list),
        np.mean(f1_px_list),
        np.mean(aupro_px_list),
    )
    print_fn(
        "Mean: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, "
        "P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}".format(*mean_metrics)
    )
    _log_metric_table(metric_rows, print_fn=print_fn)
    return mean_metrics, metric_rows


# ------------------------- 融合权重工具 -------------------------
def _parse_fusion_weights(value: str, num_groups: int, name: str):
    if value is None:
        return None
    value = value.strip()
    if not value:
        return None
    parts = [p.strip() for p in value.split(",") if p.strip()]
    weights = [float(p) for p in parts]
    if len(weights) != num_groups:
        raise ValueError(f"{name} length {len(weights)} != num_groups {num_groups}")
    return weights


def _gate_scalar_weights(en, de, gate: nn.Module, temperature: float):
    cos_loss = nn.CosineSimilarity()
    pd_list = []
    for item in range(len(en)):
        a_ = en[item].detach()
        b_ = de[item].detach()
        point_dist = 1 - cos_loss(a_, b_).unsqueeze(1)
        pd_list.append(point_dist)
    stacked = torch.cat(pd_list, dim=1)  # [B,G,H,W]
    logits = gate(stacked)
    temp = temperature if temperature > 1e-6 else 1e-6
    weights = torch.softmax(logits / temp, dim=1)
    return weights.mean(dim=(0, 2, 3))

# ------------------------- 训练主函数 -------------------------
def train(item_list):
    total_iters = args.iters
    image_size = 448
    crop_size  = 392
    setup_seed(args.seed)
    data_transform, gt_transform = get_data_transforms(image_size, crop_size)

    train_data_list, test_data_list = [], []

    # ---------- Few-shot 设置 ----------
    shots_per_class = args.shots if args.shots in (1, 2, 4) else 1

    for i, item in enumerate(item_list):
        train_path = os.path.join(args.data_path, item, 'train')
        test_path  = os.path.join(args.data_path, item)

        train_data = ImageFolder(root=train_path, transform=data_transform)

        # --- few-shot 子采样 ---
        sorted_samples = train_data.samples
        rng = random.Random(1 + i)  # 固定随机，按类偏移
        if len(sorted_samples) >= shots_per_class:
            keep_samples = rng.sample(sorted_samples, k=shots_per_class)
        else:
            keep_samples = sorted_samples
        train_data.samples = keep_samples[:]
        train_data.imgs    = keep_samples[:]

        # 重写类别索引为当前类 i
        train_data.classes = [item]
        train_data.class_to_idx = {item: i}
        train_data.samples = [(p, i) for (p, _) in train_data.samples]
        train_data.imgs    = train_data.samples[:]
        train_data.targets = [i] * len(train_data.samples)

        # 测试集
        test_data = MVTecDataset(root=test_path, transform=data_transform,
                                 gt_transform=gt_transform, phase="test")

        train_data_list.append(train_data)
        test_data_list.append(test_data)

    # 合并所有类别的 few-shot 样本
    train_data = ConcatDataset(train_data_list)

    # --- DataLoader ---
    total_train_images = len(train_data)
    batch_size = max(1, min(args.batch_size, total_train_images))
    train_dataloader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.workers,
        drop_last=False
    )

    # ========== Encoder ==========
    encoder_name = args.encoder
    encoder = encoder.load(encoder_name)

    # ★ DINOv2 reg 固定为 4（与你权重一致）
    num_reg_tokens = 4

    if 'small' in encoder_name:
        embed_dim, num_heads = 384, 6
    elif 'base' in encoder_name:
        embed_dim, num_heads = 768, 12
    elif 'large' in encoder_name:
        embed_dim, num_heads = 1024, 16
        target_layers = [4, 6, 8, 10, 12, 14, 16, 18]
    else:
        raise RuntimeError("Architecture not in small/base/large.")
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    num_groups = len(fuse_layer_encoder)
    fusion_weights_static = _parse_fusion_weights(args.fusion_weights, num_groups, "--fusion_weights")
    fusion_learn_init = _parse_fusion_weights(args.fusion_learnable_init, num_groups, "--fusion_learnable_init")

    # === decoder 工厂 ===
    def make_vit_block(dim, heads, attn_type='linear'):
        attn_class = Attention if attn_type == 'full' else LinearAttention
        return GFSRBlock(
            dim=dim,
            num_heads=heads,
            mlp_ratio=4.,
            qkv_bias=True,
            norm_layer=functools.partial(nn.LayerNorm, eps=1e-8),
            attn=attn_class
        )
# 原本的make_decoder
    # def make_decoder(kind, dim, heads, pattern=None, depth=8, num_reg_tokens=4):
    #     blocks = []
    #     if kind == 'vit':
    #         for _ in range(depth):
    #             blocks.append(make_vit_block(dim, heads))
    #     elif kind == 'mamba':
    #         for _ in range(depth):
    #             # blk = DALRBlock(embed_dim=dim, num_register_tokens=num_reg_tokens)
    #             blk = DALRBlock(
    #                     embed_dim=dim,
    #                     num_register_tokens=num_reg_tokens,
    #                     num_hss=3,
    #                     lss_kwargs=dict(
    #                         ks_cfg=((3,5),(5,7),(5,7)),   # 或者为“token 版”统一用 ks=(5,7), dilations=(1,2)
    #                         dilations=(1,1),              # 如果你用的是单层 DALRLayerBlock，而非多stage decoder
    #                         add_local3=True,
    #                         add_asym=True,
    #                         hss_learn_dir=True,
    #                     )
    #                 )
    #             blk = ResidualScale(blk, init_scale=0.2)   # ★ 残差缩放
    #             blk = PostNorm(blk, dim)                   # ★ 后归一化
    #             blocks.append(blk)
    #     else:  # hybrid
    #         pat = [s.strip().lower() for s in (pattern or 'v,m,v,m,v,m,v,m').split(',')]
    #         assert len(pat) == depth
    #         for p in pat:
    #             if p == 'm':
    #                 # blk = DALRBlock(embed_dim=dim, num_register_tokens=num_reg_tokens)
    #                 blk = DALRBlock(
    #                     embed_dim=dim,
    #                     num_register_tokens=num_reg_tokens,
    #                     num_hss=3,
    #                     lss_kwargs=dict(
    #                         ks_cfg=((3,5),(5,7),(5,7)),   # 或者为“token 版”统一用 ks=(5,7), dilations=(1,2)
    #                         dilations=(1,1),              # 如果你用的是单层 DALRLayerBlock，而非多stage decoder
    #                         add_local3=True,
    #                         add_asym=True,
    #                         hss_learn_dir=True,
    #                     )
    #                 )
    #                 blk = ResidualScale(blk, init_scale=0.2)
    #                 blk = PostNorm(blk, dim)
    #             elif p == 'v':
    #                 blk = make_vit_block(dim, heads)
    #             else:
    #                 raise ValueError(f"未知标记: {p}")
    #             blocks.append(blk)
    #     return nn.ModuleList(blocks)
   
    # 11月6号使用
    # def make_decoder(kind, dim, heads, pattern=None, depth=8, num_reg_tokens=4):
    #     blocks = []

    #     def make_mamba_block():
    #         # ✅ 单块 DALRLayerBlock 正确的参数键：ks / dilations / add_local3 / add_asym / hss_learn_dir
    #         lss_kwargs = dict(
    #             ks=(5, 7),         # 两条本地分支：5×5 + 7×7
    #             dilations=(1, 2),  # 第二条分支轻度膨胀，等效更大感受野
    #             add_local3=True,   # 额外 3×3，利好 screw/capsule 细小缺陷
    #             add_asym=True,     # 1×k / k×1 各向异性，利好 cable 方向性
    #             hss_learn_dir=True # HSS 扫描方向可学习
    #         )
    #         blk = DALRBlock(
    #             embed_dim=dim,
    #             num_register_tokens=num_reg_tokens,
    #             num_hss=3,
    #             lss_kwargs=lss_kwargs
    #         )
            
    #         blk = ResidualScale(blk, init_scale=0.2)  # ★ 残差缩放
    #         blk = PostNorm(blk, dim)                  # ★ 后归一化
    #         return blk

    #     if kind == 'vit':
    #         for _ in range(depth):
    #             blocks.append(make_vit_block(dim, heads))

    #     elif kind == 'mamba':
    #         for _ in range(depth):
    #             blocks.append(make_mamba_block())

    #     else:  # hybrid
    #         pat = [s.strip().lower() for s in (pattern or 'v,m,v,m,v,m,v,m').split(',')]
    #         assert len(pat) == depth, f"hybrid_pat 长度({len(pat)})必须等于 decode_depth({depth})"
    #         for p in pat:
    #             if p == 'm':
    #                 blocks.append(make_mamba_block())
    #             elif p == 'v':
    #                 blocks.append(make_vit_block(dim, heads))
    #             else:
    #                 raise ValueError(f"未知标记: {p}")

    #     return nn.ModuleList(blocks)
    
    def make_decoder(kind, dim, heads, pattern=None, depth=8, num_reg_tokens=4):
        blocks = []

        # 小工具：安全解析 "a,b" 字符串
        def _parse_pair(s: str, default=(5, 7)):
            try:
                vals = tuple(int(x.strip()) for x in s.split(','))
                if len(vals) != 2:
                    return default
                return vals
            except Exception:
                return default

        def make_mamba_block():
            # 1) 解析核与膨胀
            ks_pair = _parse_pair(getattr(args, 'mamba_ks', '5,7'), default=(5, 7))
            dil_pair = _parse_pair(getattr(args, 'mamba_dilations', '1,2'), default=(1, 2))

            # 2) 如果用户给了单一覆盖核（>0），用它覆盖两路
            if getattr(args, 'mamba_dw_kernel', 0) and args.mamba_dw_kernel > 0:
                ks_pair = (int(args.mamba_dw_kernel), int(args.mamba_dw_kernel))

            lss_kwargs = dict(
                ks=ks_pair,                            # 例如 (3,3) 或 (5,7)
                dilations=dil_pair,                    # 例如 (1,2)
                add_local3=bool(args.mamba_add_local3),
                add_asym=bool(args.mamba_add_asym),
                hss_learn_dir=True
            )

            blk = DALRBlock(
                embed_dim=dim,
                num_register_tokens=num_reg_tokens,
                num_hss=int(args.mamba_num_hss),
                scan_method=args.mamba_scan_method,
                num_scan_dirs=args.mamba_scan_dirs,
                lss_kwargs=lss_kwargs
            )

            blk = ResidualScale(blk, init_scale=0.2)  # ★ 残差缩放
            blk = PostNorm(blk, dim)                  # ★ 后归一化
            return blk

        if kind == 'vit':
            for _ in range(depth):
                blocks.append(make_vit_block(dim, heads, args.decoder_attn))

        elif kind == 'mamba':
            for _ in range(depth):
                blocks.append(make_mamba_block())

        else:  # hybrid
            pat = [s.strip().lower() for s in (pattern or 'v,m,v,m,v,m,v,m').split(',')]
            assert len(pat) == depth, f"hybrid_pat 长度({len(pat)})必须等于 decode_depth({depth})"
            for p in pat:
                if p == 'm':
                    blocks.append(make_mamba_block())
                elif p == 'v':
                    blocks.append(make_vit_block(dim, heads, args.decoder_attn))
                else:
                    raise ValueError(f"未知标记: {p}")

        return nn.ModuleList(blocks)


    # --------- Bottleneck 选择器（按命令行选择） ---------
    def make_bottleneck_module(embed_dim, variant: str):
        # --- 新增：如果 variant 为 'none'，则返回一个恒等模块 ---
        if variant == 'none':
            # print_fn is not available here, so we use a standard print
            print("[bottleneck] variant=none (bottleneck layer is REMOVED)")
            return nn.Identity()

        hf = int(2.67 * embed_dim)  # 默认宽度
        common = dict(drop=args.bn_drop, grad=0.7)

        if variant == 'cgb':
            return CGB(embed_dim, hf, embed_dim, **common)

        elif variant == 'noisy_geglu':
            return NoisyGEGLUBottleneck(
                in_features=embed_dim,
                hidden_features=int(8 * embed_dim / 3),  # 贴合你之前 2048 的宽度
                out_features=embed_dim,
                dropout_schedule=DropoutSchedule(
                    start_p=args.bn_drop_start, end_p=args.bn_drop_end, warmup_steps=args.bn_drop_warmup
                ),
                detach_ratio=0.0
            )

        elif variant == 'jitter':
            # 训练态加性高斯噪声；Eval 自动关闭
            return FeatureJitter(sigma=args.jitter_sigma, p=1.0)

        elif variant == 'mask':
            return MaskTokenBottleneck(dim=embed_dim, mask_ratio=args.mask_ratio, learnable=True)
        elif variant == 'rsc':
                return RSCReconstructionHead(
                    input_dim=embed_dim,
                    output_dim=embed_dim,  # 与 G2LHyD 架构兼容
                    dropout=args.bn_drop   # 复用现有 dropout 参数
                )
        elif variant == 'hvq':
            # 用 wrapper 在前向里抓取 loss（保持 G2LHyD 接口只收/发 [B,N,C] 张量）
            class _HVQWrapper(nn.Module):
                def __init__(self, dim, k, beta, ema_decay):
                    super().__init__()
                    self.hvq = HVQBottle(dim=dim, K=k, beta=beta, ema_decay=ema_decay)
                    self.last_vq_loss = None
                def forward(self, x):
                    z, loss_vq = self.hvq(x)
                    self.last_vq_loss = loss_vq
                    return z
        
            return _HVQWrapper(embed_dim, args.vq_k, args.vq_beta, args.vq_ema_decay)
        elif variant == 'gmlp_dw':
            # drop/grad 建议 0.1 / 0.7；kernel 用 args.bn_dw_kernel（默认5）
            return SAB(
                in_features=embed_dim, hidden_features=int(2 * embed_dim), out_features=embed_dim,
                drop=args.bn_drop, grad=0.7, dw_kernel=args.bn_dw_kernel,
                num_register_tokens=num_reg_tokens, has_cls=True
            )

        elif variant == 'gmlp_lin':
            # 推荐：默认用 banded，带宽=7（更稳）；若想保持纯 linear，就把 token_mode='linear'
            return LinearGMLPTokenBlock(
                in_features=embed_dim, hidden_features=int(2 * embed_dim), out_features=embed_dim,
                drop=args.bn_drop, grad=0.7,
                num_register_tokens=num_reg_tokens, has_cls=True,
                token_mode='banded',  # 改成 'linear' 或 'lowrank' 也行
                bandwidth=7,          # 只在 banded 模式下生效
                rank=64               # 只在 lowrank 模式下生效
            )

        else:
            raise ValueError(f'未知瓶颈 variant: {variant}')


    bottleneck = nn.ModuleList([ make_bottleneck_module(embed_dim, args.bottleneck_variant) ])

    decoder = make_decoder(args.decoder, embed_dim, num_heads,
                           pattern=args.hybrid_pat, depth=args.decode_depth,
                           num_reg_tokens=num_reg_tokens)
    print_fn(f"[decoder] using: {args.decoder} | "
             f"{'pattern=' + args.hybrid_pat if args.decoder=='hybrid' else 'pure'} | "
             f"depth={args.decode_depth} | num_reg_tokens={num_reg_tokens}")
    print_fn(f"[bottleneck] variant={args.bottleneck_variant}")

    # ========== Model ==========
    model = G2LHyD(
        encoder=encoder,
        bottleneck=bottleneck,
        decoder=decoder,
        target_layers=target_layers,
        mask_neighbor_size=0,
        fuse_layer_encoder=fuse_layer_encoder,
        fuse_layer_decoder=fuse_layer_decoder
    )
    model = model.to(device)

    fusion_weight_module = None
    fusion_gate = None
    if args.fusion_mode == "learnable":
        fusion_weight_module = LearnableFusionWeights(
            num_groups=num_groups,
            temperature=args.fusion_temp,
            init_weights=fusion_learn_init,
        )
    elif args.fusion_mode == "spatial_gate":
        fusion_gate = SpatialFusionGate(num_groups=num_groups, hidden=args.fusion_gate_hidden)
    if fusion_weight_module is not None:
        fusion_weight_module = fusion_weight_module.to(device)
        model.fusion_weight_module = fusion_weight_module
    if fusion_gate is not None:
        fusion_gate = fusion_gate.to(device)
        model.fusion_gate = fusion_gate

    def _current_fusion_weights():
        if args.fusion_mode == "weighted_mean":
            return fusion_weights_static
        if args.fusion_mode == "learnable" and fusion_weight_module is not None:
            return fusion_weight_module().detach()
        return None

    # ========== 可选：加载已有权重 / 仅可视化 ==========
    if args.load_ckpt:
        ckpt_path = os.path.expanduser(args.load_ckpt)
        # weights_only=True 为默认，但有些旧 ckpt 需要完整反序列化
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = _extract_state_dict(state)
        missing, unexpected, mismatched = _load_state_dict_with_fallback(
            model, state_dict, print_fn=print_fn
        )
        print_fn(
            f"[ckpt] loaded from {ckpt_path} | missing={len(missing)} | "
            f"unexpected={len(unexpected)} | mismatched={len(mismatched)}"
        )
        if mismatched and any("dir_logits" in k for k, _, _ in mismatched):
            print_fn("[ckpt] tip: this checkpoint likely used --mamba_scan_dirs 2")

    if args.visualize_only:
        fusion_weights_eval = _current_fusion_weights()
        evaluate_mvtec_metrics(
            model=model,
            item_list=item_list,
            test_data_list=test_data_list,
            train_data_list=train_data_list,
            device=device,
            batch_size=batch_size,
            workers=args.workers,
            args=args,
            fusion_weights=fusion_weights_eval,
            fusion_gate=fusion_gate,
            print_fn=print_fn,
        )
        visualize_mvtec_results(
            model=model,
            test_data_list=test_data_list,
            item_list=item_list,
            device=device,
            workers=args.workers,
            save_root=args.vis_root,
            max_images=args.vis_limit,
            vis_alpha=args.visualize_alpha,
            vis_sigma=args.visualize_sigma,
            fusion_mode=args.fusion_mode,
            fusion_confidence=args.fusion_confidence,
            fusion_topk=args.fusion_topk,
            fusion_temp=args.fusion_temp,
            fusion_weights=fusion_weights_eval,
            fusion_gate=fusion_gate,
            print_fn=print_fn
        )
        return

    # ========== Init ==========
    trainable_modules = [bottleneck, decoder]
    if fusion_weight_module is not None:
        trainable_modules.append(fusion_weight_module)
    if fusion_gate is not None:
        trainable_modules.append(fusion_gate)
    trainable = nn.ModuleList(trainable_modules)
    for m in trainable.modules():
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.01, a=-0.03, b=0.03)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # ========== Optim & LR ==========
    lr = args.lr
    warmup = args.warmup
    if args.decoder in ('mamba', 'hybrid'):
        lr = lr * 0.5            # 避免过更新
        warmup = max(warmup, 500)
    optimizer = StableAdamW([{'params': trainable.parameters()}],
                            lr=lr, betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=True, eps=1e-10)
    lr_scheduler = WarmCosineScheduler(
        optimizer, base_value=lr, final_value=args.lr_end, total_iters=total_iters, warmup_iters=warmup
    )

    print_fn('train image number: {}'.format(len(train_data)))
    print_fn(f'few-shot setting: {shots_per_class} per class')

    it = 0
    epoch_len = max(1, len(train_dataloader))
    for epoch in range(int(np.ceil(total_iters / epoch_len))):
        model.train()
        if fusion_weight_module is not None:
            fusion_weight_module.train()
        if fusion_gate is not None:
            fusion_gate.train()
        loss_list = []
        for img, label in train_dataloader:
            img = img.to(device)
            label = label.to(device)

            en, de = model(img)

            p_final = 0.7 if args.decoder in ('mamba', 'hybrid') else 0.9
            p = min(p_final * it / 3000, p_final)  # 3k iter 再拉满

            loss_weights = None
            if fusion_weight_module is not None:
                loss_weights = fusion_weight_module()
            elif fusion_gate is not None:
                loss_weights = _gate_scalar_weights(en, de, fusion_gate, args.fusion_temp)

            loss = global_cosine_hm_percent(en, de, p=p, factor=0.1, weights=loss_weights)


            # ★ 如果用了 hvq，把瓶颈里 wrapper 暂存的损失加进来
            vq_extra = 0.0
            for m in bottleneck.modules():
                if hasattr(m, "last_vq_loss") and (m.last_vq_loss is not None):
                    vq_extra = vq_extra + m.last_vq_loss
            if isinstance(vq_extra, torch.Tensor):
                loss = loss + args.vq_lambda * vq_extra


            optimizer.zero_grad()
            # ★ Dropout 预热：遍历瓶颈里支持 step_scheduler 的模块
            for m in bottleneck.modules():
                if hasattr(m, "step_scheduler"):
                    m.step_scheduler(1)
            
            loss.backward()
            nn.utils.clip_grad_norm_(trainable.parameters(), max_norm=0.1)
            optimizer.step()
            loss_list.append(loss.item())
            lr_scheduler.step()

           # --- 定期评测 & 日志：先自增，再统一用 it（自然数） ---
            it += 1

            # 评测触发：从 eval_start 开始，每隔 eval_every 次评一次
            if (it % args.log_every) == 0:
                print_fn('iter [{}/{}], loss:{:.4f}'.format(it, total_iters, np.mean(loss_list)))
                loss_list = []

            if (it >= args.eval_start) and ((it - args.eval_start) % args.eval_every == 0):
                print_fn(f'======== EVAL @ step {it} ========')
                if fusion_weight_module is not None:
                    fusion_weight_module.eval()
                if fusion_gate is not None:
                    fusion_gate.eval()
                fusion_weights_eval = _current_fusion_weights()
                mean_metrics, _ = evaluate_mvtec_metrics(
                    model=model,
                    item_list=item_list,
                    test_data_list=test_data_list,
                    train_data_list=train_data_list,
                    device=device,
                    batch_size=batch_size,
                    workers=args.workers,
                    args=args,
                    fusion_weights=fusion_weights_eval,
                    fusion_gate=fusion_gate,
                    print_fn=print_fn,
                )

                if args.save_eval_ckpt:
                    ckpt_dir = os.path.join(args.save_dir, args.save_name, "ckpts_eval")
                    os.makedirs(ckpt_dir, exist_ok=True)
                    ckpt_path = os.path.join(ckpt_dir, f"iter_{it:06d}.pth")
                    torch.save({'iter': it, 'model': model.state_dict(), 'metrics': mean_metrics}, ckpt_path)
                    print_fn(f"[ckpt_eval] saved to {ckpt_path}")

                model.train()
                if fusion_weight_module is not None:
                    fusion_weight_module.train()
                if fusion_gate is not None:
                    fusion_gate.train()

            # # 训练日志：同样用 it，自然对齐
            # if (it % args.log_every) == 0:
            #     print_fn('iter [{}/{}], loss:{:.4f}'.format(it, total_iters, np.mean(loss_list)))
            #     loss_list = []
            
            # 终止条件
            if it == total_iters:
                break

    if args.save_visuals or args.visualize:
        visualize_mvtec_results(
            model=model,
            test_data_list=test_data_list,
            item_list=item_list,
            device=device,
            workers=args.workers,
            save_root=args.vis_root,
            max_images=args.vis_limit,
            vis_alpha=args.visualize_alpha,
            vis_sigma=args.visualize_sigma,
            fusion_mode=args.fusion_mode,
            fusion_confidence=args.fusion_confidence,
            fusion_topk=args.fusion_topk,
            fusion_temp=args.fusion_temp,
            fusion_weights=_current_fusion_weights(),
            fusion_gate=fusion_gate,
            print_fn=print_fn
        )

    if args.export_ckpt:
        ckpt_path = os.path.expanduser(args.export_ckpt)
        ckpt_dir = os.path.dirname(ckpt_path)
        if ckpt_dir:
            os.makedirs(ckpt_dir, exist_ok=True)
        torch.save(model.state_dict(), ckpt_path)
        print_fn(f"[ckpt] exported to {ckpt_path}")

    return


# ------------------------- 入口 -------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Few-shot Dinomaly with Mamba/Vit/Hybrid decoder (MVTEC)')
    # 数据路径与保存
    parser.add_argument('--data_path', type=str,
                        default='/home/hyz/MyDisk/hyz/Dinomaly/data/mvtec_anomaly_detection/')
    parser.add_argument('--save_dir',  type=str,
                        default='/home/hyz/MyDisk/hyz/Dinomaly/data/final/mvtec/zhu/')
    parser.add_argument('--save_name', type=str, default='bn_none_seed1')
    parser.add_argument('--export_ckpt', type=str, default='',
                        help='路径不为空时，训练结束后将模型 state_dict 导出到该 ckpt 文件')
    parser.add_argument('--load_ckpt', type=str, default='',
                        help='若指定则在训练/可视化前加载该 ckpt')
    parser.add_argument('--visualize_only', action='store_true',
                        help='只跑可视化与像素 AUROC，跳过训练')
    parser.add_argument('--visualize', action='store_true',
                        help='推理阶段导出 original/heatmap/overlay 三类图')
    parser.add_argument('--save_visuals', action='store_true',
                        help='训练完成后跑可视化与像素 AUROC')
    parser.add_argument('--vis_root', type=str, default='results',
                        help='可视化结果保存的根目录（默认 results/<class>/imgs）')
    parser.add_argument('--vis_limit', type=int, default=0,
                        help='每类最多保存的可视化样本数，0 表示不限制')
    parser.add_argument('--visualize_sigma', type=float, default=4.0,
                        help='可视化热力图高斯平滑 sigma，默认 4（与 One-For-All 风格一致）')
    parser.add_argument('--visualize_alpha', type=float, default=0.5,
                        help='overlay 融合系数 alpha（0~1，值越大原图占比越高），如 0.8')

    # few-shot & 设备 & 资源
    parser.add_argument('--shots', type=int, default=1, choices=[1, 2, 4],
                        help='few-shot per class')
    parser.add_argument('--seed', type=int, default=1, help='随机种子')
    parser.add_argument('--cuda', type=int, default=0, help='CUDA device index')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=4)
    # 解码器与深度
    parser.add_argument('--decoder', type=str, default='hybrid',
                        choices=['vit', 'mamba', 'hybrid'],
                        help='decoder type: pure ViT / pure Mamba / hybrid (Mamba↔ViT)')
    parser.add_argument('--decoder_attn', type=str, default='linear', choices=['linear', 'full'],
                        help='attention type for ViT blocks in decoder')
    parser.add_argument('--hybrid_pat', type=str, default='v,v,v,v,m,m,m,m',
                        help='pattern for hybrid across N blocks, m= Mamba, v= ViT')
    parser.add_argument('--decode_depth', type=int, default=8,
                        help='number of decoder blocks (default 8 to match your original)')

    # 编码器选择
    parser.add_argument('--encoder', type=str, default='dinov2reg_vit_base_14',
                        help='{dinov2reg_vit_small_14 | dinov2reg_vit_base_14 | dinov2reg_vit_large_14 | ...}')

    parser.add_argument("--bn_mlp_ratio", type=float, default=4.0)
    parser.add_argument("--bn_dw_kernel", type=int, default=5)
    parser.add_argument("--bn_drop", type=float, default=0.1)

    # 优化器 & 学习率调度
    parser.add_argument('--lr', type=float, default=2e-3, help='learning rate')
    parser.add_argument('--lr_end', type=float, default=2e-4, help='end learning rate for warm-cosine scheduler')
    parser.add_argument('--warmup', type=int, default=100, help='warmup iterations')
    
    # 日志与评测频率
    parser.add_argument('--eval_start', type=int, default=10000,
                    help='first evaluation step; start evaluating at this step, then every eval_every steps')

    parser.add_argument('--eval_every', type=int, default=5000)
    parser.add_argument('--log_every',  type=int, default=500)
    parser.add_argument('--save_eval_ckpt', type=int, default=1, choices=[0, 1],
                        help='if 1, save checkpoint at each evaluation iteration')
    # ===== Step-1 / Step-2 开关与超参（仅评测期生效） =====
    parser.add_argument('--hotspot_gem', action='store_true',
                        help='use Hotspot-GeM aggregator for image-level scores')
    parser.add_argument('--topk_ratio', type=float, default=0.032,
                        help='top-k ratio for hotspot selection, e.g., 0.02=2%%')
    parser.add_argument('--gem_p', type=float, default=6.0,
                        help='GeM pooling p for image-level aggregation')
    parser.add_argument('--fg_q', type=int, default=64,
                        help='foreground gate quantile on token norm (0-100)')
    parser.add_argument('--flip_tta', action='store_true',
                        help='use horizontal flip TTA at eval time')
    parser.add_argument('--z_norm', action='store_true',
                        help='class-wise z-normalization for image scores at eval')
    parser.add_argument('--postproc', action='store_true',
                        help='morphological smoothing on heatmap at eval (Step-2)')
    parser.add_argument('--post_k', type=int, default=3, help='morph kernel size (odd)')
    parser.add_argument('--post_iters', type=int, default=1, help='morph iterations')
    parser.add_argument('--rot_tta', type=str, default='none',
                        help="Rotation TTA policy: 'none'|'all'|'textures' or comma-separated class list")
    parser.add_argument('--ms_tta', type=str, default='1.0,1.10',
                        help="Multi-scale TTA factors, e.g. '0.75,1.0' (1.0 is implicit and will be skipped)")
    parser.add_argument('--tex_rot_tta', action='store_true',
                        help='Enable rotation TTA only for texture classes (carpet/grid/leather/tile/wood).')
    parser.add_argument('--i_agg', type=str, default='max', choices=['max', 'max_topk'],
                        help='Image-level aggregator: max (default) or max_topk')
    parser.add_argument('--i_alpha', type=float, default=0.6,
                        help='alpha for max_topk aggregator (0..1)')
    parser.add_argument('--i_top_percent', type=float, default=5.0,
                        help='top-percent for max_topk aggregator (e.g. 5 means top-5%% pixels)')

    # --- 融合策略 ---
    parser.add_argument('--fusion_mode', type=str, default='mean',
                        choices=['mean', 'max', 'weighted_mean', 'conf_softmax', 'learnable', 'spatial_gate'],
                        help='fusion over groups: mean/max/weighted_mean/conf_softmax/learnable/spatial_gate')
    parser.add_argument('--fusion_weights', type=str, default='',
                        help='comma weights for weighted_mean, e.g., "0.6,0.4"')
    parser.add_argument('--fusion_confidence', type=str, default='topk_mean',
                        choices=['topk_mean', 'variance', 'entropy', 'mean'],
                        help='confidence type for conf_softmax')
    parser.add_argument('--fusion_topk', type=float, default=0.02,
                        help='top-k ratio for conf_softmax')
    parser.add_argument('--fusion_temp', type=float, default=0.07,
                        help='softmax temperature for conf_softmax/learnable/spatial_gate')
    parser.add_argument('--fusion_learnable_init', type=str, default='',
                        help='init weights for learnable, e.g., "0.5,0.5"')
    parser.add_argument('--fusion_gate_hidden', type=int, default=16,
                        help='hidden channels for spatial_gate')

    # # ==== 瓶颈变体选择 ====
    # parser.add_argument('--bottleneck_variant', type=str, default='cgb',
    #                     choices=['cgb', 'pg_cgb', 'cgb_lora', 'ms', 'pixel', 'cgbv2'],
    #                     help='原始 cgb / PG-CGB / CGB-LoRA / 简化多尺度 / PixelOptimized / CGBv2')
    parser.add_argument('--bottleneck_variant', type=str, default='gmlp_dw',
        choices=['none', 'cgb', 'pg_cgb', 'cgb_lora', 'ms', 'pixel', 'cgbv2', 'osb', 'sgb','gmlp_dw','gmlp_lin',
                # ★ 新增四种
                'noisy_geglu', 'jitter', 'mask', 'hvq'  # 如需 hfpn 请先改 G2LHyD
        ],
        help='瓶颈：CGB 系列 / OSB / SGB / + 新增 noisy_geglu | jitter | mask | hvq')
    # --- Noisy-GEGLU（bMLP）Dropout 预热 ---
    parser.add_argument('--bn_drop_start',  type=float, default=0.0)
    parser.add_argument('--bn_drop_end',    type=float, default=0.2)
    parser.add_argument('--bn_drop_warmup', type=int,   default=1000)

    # --- Jitter / Mask ---
    parser.add_argument('--jitter_sigma', type=float, default=0.10)
    parser.add_argument('--mask_ratio',   type=float, default=0.30)

    # --- HVQ ---
    parser.add_argument('--vq_k',         type=int,   default=512)
    parser.add_argument('--vq_beta',      type=float, default=0.25)
    parser.add_argument('--vq_ema_decay', type=float, default=0.99)
    parser.add_argument('--vq_lambda',    type=float, default=0.25, help='loss 权重')

    # --- Mamba（解码块里的本地卷积/HSS）---
    parser.add_argument('--mamba_ks', type=str, default='5,7',
                        help="Mamba LSS 的两个本地卷积分支核，如 '5,7'")
    parser.add_argument('--mamba_dilations', type=str, default='1,2',
                        help="对应两个分支的膨胀系数，如 '1,2'")
    parser.add_argument('--mamba_dw_kernel', type=int, default=0,
                        help='若 >0 则两路核都设为该值，如 3 -> ks=(3,3)')
    parser.add_argument('--mamba_add_local3', type=int, default=1, choices=[0,1],
                        help='是否额外加一条 3x3 分支（1=开，0=关）')
    parser.add_argument('--mamba_add_asym', type=int, default=1, choices=[0,1],
                        help='是否加入 1xk / kx1 非对称卷积（1=开，0=关）')
    parser.add_argument('--mamba_num_hss', type=int, default=3,
                        help='HSS 扫描分支条数（默认 3）')

    # +++ Mamba 扫描方式与方向 +++
    parser.add_argument('--mamba_scan_method', type=str, default='hilbert',
                        choices=['hilbert', 'sweep', 'scan'],
                        help="Scan method for HSS module in Mamba (hilbert/sweep/scan).")
    parser.add_argument('--mamba_scan_dirs', type=int, default=8,
                        choices=[2, 4, 8],
                        help="Number of scan directions for HSS module (2/4/8).")

    parser.add_argument('--obj', type=str, default=None,
                        help='Specific object class to test (e.g., bottle). If None, test all classes.')
    parser.add_argument('--iters', type=int, default=30000,
                        help='Number of training iterations.')


    args = parser.parse_args()

    # 类别列表（MVTEC 15 类）
    item_list = [
        'carpet', 'grid', 'leather', 'tile', 'wood',
        'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut',
        'pill', 'screw', 'toothbrush', 'transistor', 'zipper'
    ]
    if args.obj:
        if args.obj in item_list:
            item_list = [args.obj]
        else:
            raise ValueError(f"Object {args.obj} not found in MVTec classes.")

    # Logger
    logger   = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info

    # 设备（按命令行选择）
    device = f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu'
    print_fn(device)

    # 训练/评测
    train(item_list)
