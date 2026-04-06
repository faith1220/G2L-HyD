
# Few-shot Dinomaly (ViT encoder + 可选 Mamba / ViT / Hybrid token解码)
# - 默认 1-shot
# - 默认解码器 mamba
# - 训练/评测逻辑保持你的原样：evaluation_batch(), global_cosine_hm_percent 等
# - 已接 Step-A（Hotspot-GeM + 前景门控）/ Step-B（Flip-TTA + 类内 z-norm）在评测期使用

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
    SAB, ConvBlock, FeatureJitter, CGB,
    NoisyGEGLUBottleneck, MaskTokenBottleneck, HFPNBottleneck, HVQBottle, RSCReconstructionHead
)
from dalr.fewshot_g2l import FewShotG2L   # ← 新增

from utils import (
    evaluation_batch, global_cosine, regional_cosine_hm_percent,
    global_cosine_hm_percent, WarmCosineScheduler,
    compute_image_score_from_heatmap, cal_anomaly_maps,
    _should_rotate, _parse_scales, _morph_smooth
)
from visualization import visualizer
from ptflops import get_model_complexity_info
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
    return {(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()}


def _extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        if "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
            return ckpt_obj["model"]
        if "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
            return ckpt_obj["state_dict"]
    return ckpt_obj


def _load_state_dict_with_fallback(model, state_dict, print_fn=print):
    """
    strict=False 在 shape mismatch 时仍会报错，这里只加载 shape 匹配的权重。
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


def visualize_visa_results(
    model,
    test_data_list,
    item_list,
    device,
    workers: int = 0,
    save_root: str = "results_visa",
    max_images: int = 0,
    vis_alpha: float = 0.5,
    vis_sigma: float = 4.0,
    # 评测一致性参数
    flip_tta: bool = False,
    rot_tta: str = "none",
    ms_tta: str = "",
    resize_mask: int | None = 256,
    postproc: bool = False,
    post_k: int = 3,
    post_iters: int = 1,
    print_fn=print,
):
    """
    按类别导出三类图：
    <save_root>/imgs/<class>/<defect>/{original,heatmap,overlay}/filename
    """
    model.eval()
    scales = _parse_scales(ms_tta)

    # 读取 patch_size 用于多尺度 TTA 对齐
    patch_h = patch_w = 14
    pe = getattr(model, 'encoder', None)
    pe = getattr(pe, 'patch_embed', None) if pe is not None else None
    ps = getattr(pe, 'patch_size', None) if pe is not None else None
    if isinstance(ps, (tuple, list)) and len(ps) == 2:
        patch_h, patch_w = int(ps[0]), int(ps[1])
    elif isinstance(ps, int):
        patch_h = patch_w = int(ps)

    @torch.no_grad()
    def _resize_to_patch_multiple(x, scale: float):
        H, W = x.shape[-2], x.shape[-1]
        Ht = int(round(H * scale))
        Wt = int(round(W * scale))
        Ht = max(patch_h, int(round(Ht / patch_h)) * patch_h)
        Wt = max(patch_w, int(round(Wt / patch_w)) * patch_w)
        return F.interpolate(x, size=(Ht, Wt), mode='bilinear', align_corners=False)

    @torch.no_grad()
    def _fwd_heatmap(x):
        en, de = model(x)
        amap, _ = cal_anomaly_maps(en, de, x.shape[-1])  # [B,?,H,W]
        return amap

    @torch.no_grad()
    def _tta_merge(img, cls_name: str):
        H, W = img.shape[-2], img.shape[-1]
        maps = []

        # 原图
        amap = _fwd_heatmap(img)
        maps.append(amap)

        # 水平翻转
        if flip_tta:
            x = torch.flip(img, dims=[-1])
            a = _fwd_heatmap(x)
            a = torch.flip(a, dims=[-1])
            maps.append(a)

        # 旋转
        if _should_rotate(cls_name, rot_tta):
            for k in (1, 2, 3):
                x = torch.rot90(img, k=k, dims=[-2, -1])
                a = _fwd_heatmap(x)
                a = torch.rot90(a, k=4 - k, dims=[-2, -1])
                maps.append(a)

        # 多尺度
        for s in scales:
            if abs(float(s) - 1.0) < 1e-6:
                continue
            x_s = _resize_to_patch_multiple(img, float(s))
            a = _fwd_heatmap(x_s)
            a = F.interpolate(a, size=(H, W), mode='bilinear', align_corners=False)
            maps.append(a)

        out = maps[0]
        for m in maps[1:]:
            out = torch.maximum(out, m)
        return out  # [B,?,H,W]

    with torch.no_grad():
        for item, dataset in zip(item_list, test_data_list):
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=1, shuffle=False, num_workers=workers
            )

            for idx, (img, _, _, img_path) in enumerate(loader):
                img = img.to(device)
                anomaly_map = _tta_merge(img, cls_name=item)

                target_h, target_w = img.shape[-2], img.shape[-1]

                # resize -> 平滑 -> morph
                if resize_mask is not None:
                    if isinstance(resize_mask, int):
                        anomaly_map = F.interpolate(
                            anomaly_map, size=(resize_mask, resize_mask), mode='bilinear', align_corners=False
                        )
                    else:
                        anomaly_map = F.interpolate(
                            anomaly_map, size=resize_mask, mode='bilinear', align_corners=False
                        )
                if anomaly_map.shape[-2:] != (target_h, target_w):
                    anomaly_map = F.interpolate(
                        anomaly_map, size=(target_h, target_w), mode='bilinear', align_corners=False
                    )
                if postproc:
                    anomaly_map = _morph_smooth(anomaly_map, k=post_k, iters=post_iters)
                if anomaly_map.shape[1] > 1:
                    anomaly_map = anomaly_map.amax(dim=1, keepdim=True)

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


def evaluate_visa_metrics(
    model,
    item_list,
    test_data_list,
    train_data_list,
    device,
    batch_size,
    workers,
    args,
    print_fn=print,
):
    agg_fn = None
    if args.hotspot_gem:
        agg_fn = partial(
            compute_image_score_from_heatmap, k_ratio=args.topk_ratio, p=args.gem_p, fg_q=args.fg_q
        )
    elif args.i_agg == "max_topk":
        from utils import image_score_max_topk

        agg_fn = partial(image_score_max_topk, alpha=args.i_alpha, top_percent=args.i_top_percent)

    model.eval()
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

# ------------------------- 训练主函数 -------------------------
def train(item_list):
    total_iters = 20000
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

    # ★ 按你的要求，reg 固定为 4
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
            ks_pair = _parse_pair(getattr(args, 'mamba_ks', '5,7'), default=(5, 7))
            dil_pair = _parse_pair(getattr(args, 'mamba_dilations', '1,2'), default=(1, 2))
            if getattr(args, 'mamba_dw_kernel', 0) and args.mamba_dw_kernel > 0:
                ks_pair = (int(args.mamba_dw_kernel), int(args.mamba_dw_kernel))

            lss_kwargs = dict(
                ks=ks_pair,
                dilations=dil_pair,
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

    # ========== Bottleneck & Decoder ==========
    # bottleneck = nn.ModuleList([
    #     CGB(embed_dim, embed_dim * 4, embed_dim, drop=0.2)
    # ])
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

    # ========== 可选：加载已有权重 / 仅可视化 ==========
    if getattr(args, "load_ckpt", None):
        ckpt_path = os.path.expanduser(args.load_ckpt)
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

    if getattr(args, "visualize_only", False):
        evaluate_visa_metrics(
            model=model,
            item_list=item_list,
            test_data_list=test_data_list,
            train_data_list=train_data_list,
            device=device,
            batch_size=batch_size,
            workers=args.workers,
            args=args,
            print_fn=print_fn,
        )
        visualize_visa_results(
            model=model,
            test_data_list=test_data_list,
            item_list=item_list,
            device=device,
            workers=args.workers,
            save_root=args.vis_root,
            max_images=args.vis_limit,
            vis_alpha=args.visualize_alpha,
            vis_sigma=args.visualize_sigma,
            flip_tta=args.flip_tta if args.visualize_use_tta else False,
            rot_tta=args.rot_tta if args.visualize_use_tta else "none",
            ms_tta=args.ms_tta if args.visualize_use_tta else "1.0",
            resize_mask=(args.vis_resize_mask if args.vis_resize_mask > 0 else None),
            postproc=args.postproc if args.visualize_use_tta else False,
            post_k=args.post_k,
            post_iters=args.post_iters,
            print_fn=print_fn
        )
        return

    # ========== Init ==========
    trainable = nn.ModuleList([bottleneck, decoder])
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
        lr = lr * 0.5            # → 1e-3，避免过更新
        warmup = max(warmup, 500)  # 更长热身，减小早期漂移
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
        loss_list = []
        for img, label in train_dataloader:
            img = img.to(device)
            label = label.to(device)

            en, de = model(img)

            p_final = 0.7 if args.decoder in ('mamba', 'hybrid') else 0.9
            p = min(p_final * it / 3000, p_final)  # 3k iter 再拉满

            loss = global_cosine_hm_percent(en, de, p=p, factor=0.1)

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

            # --- 定期评测 ---
            if (it + 1) % args.eval_every == 0:
                evaluate_visa_metrics(
                    model=model,
                    item_list=item_list,
                    test_data_list=test_data_list,
                    train_data_list=train_data_list,
                    device=device,
                    batch_size=batch_size,
                    workers=args.workers,
                    args=args,
                    print_fn=print_fn,
                )
                model.train()


            it += 1
            if it == total_iters:
                break

            if (it + 1) % args.log_every == 0:
                print_fn('iter [{}/{}], loss:{:.4f}'.format(it, total_iters, np.mean(loss_list)))
                loss_list = []

    # ========== 训练结束：保存模型 & 可选跑可视化 ==========
    ckpt_dir = os.path.join(args.save_dir, args.save_name, "ckpts_final")
    os.makedirs(ckpt_dir, exist_ok=True)
    final_ckpt_path = args.export_ckpt or os.path.join(ckpt_dir, "final.pth")
    torch.save(model.state_dict(), final_ckpt_path)
    print_fn(f"[ckpt] saved to {final_ckpt_path}")

    if getattr(args, "save_visuals", False) or getattr(args, "visualize", False):
        visualize_visa_results(
            model=model,
            test_data_list=test_data_list,
            item_list=item_list,
            device=device,
            workers=args.workers,
            save_root=args.vis_root,
            max_images=args.vis_limit,
            vis_alpha=args.visualize_alpha,
            vis_sigma=args.visualize_sigma,
            flip_tta=args.flip_tta if args.visualize_use_tta else False,
            rot_tta=args.rot_tta if args.visualize_use_tta else "none",
            ms_tta=args.ms_tta if args.visualize_use_tta else "1.0",
            resize_mask=(args.vis_resize_mask if args.vis_resize_mask > 0 else None),
            postproc=args.postproc if args.visualize_use_tta else False,
            post_k=args.post_k,
            post_iters=args.post_iters,
            print_fn=print_fn
        )

    return

# ------------------------- 入口 -------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Few-shot Dinomaly with Mamba/Vit/Hybrid decoder (MVTEC)')
    # 数据路径与保存
    # 路径 & 命名
    parser.add_argument('--data_path', type=str, default='/home/user/project/Dinomaly/data/VisA_pytorch/1cls')
    parser.add_argument('--save_dir',  type=str, default='/home/user/project/Dinomaly/data/final/visa/zhu/')
    parser.add_argument('--save_name', type=str, default='visa_fewshot_mamba_geglu_k1')
    parser.add_argument('--export_ckpt', type=str, default='',
                        help='若提供则将最终模型保存到此路径，否则保存到 save_dir/save_name/ckpts_final/final.pth')
    parser.add_argument('--load_ckpt', type=str, default='',
                        help='若指定则在训练/可视化前加载该 ckpt')
    parser.add_argument('--visualize_only', action='store_true',
                        help='仅跑可视化与像素 AUROC，跳过训练')
    parser.add_argument('--visualize', action='store_true',
                        help='推理阶段导出 original/heatmap/overlay 三类图')
    parser.add_argument('--save_visuals', action='store_true',
                        help='训练完成后跑可视化并保存')
    parser.add_argument('--vis_root', type=str, default='results_visa',
                        help='可视化输出根目录（默认 results_visa/<class>/...）')
    parser.add_argument('--vis_limit', type=int, default=0,
                        help='每类最多保存多少张可视化，0 表示不限制')
    parser.add_argument('--visualize_sigma', type=float, default=4.0,
                        help='可视化热力图高斯平滑 sigma，默认 4')
    parser.add_argument('--visualize_alpha', type=float, default=0.5,
                        help='overlay 融合系数 alpha（0~1，值越大原图占比越高）')
    parser.add_argument('--vis_resize_mask', type=int, default=0,
                        help='可视化时先缩放 anomaly_map 到该尺寸，0 表示不缩放（与 MVTec OFA 风格一致）')
    parser.add_argument('--visualize_use_tta', action='store_true',
                        help='可视化时启用评测 TTA（默认关闭，保证与 MVTec OFA 风格一致）')

    # few-shot & 设备 & 资源
    parser.add_argument('--shots', type=int, default=4, choices=[1, 2, 4],
                        help='few-shot per class')
    parser.add_argument('--seed', type=int, default=5, help='随机种子')
    parser.add_argument('--cuda', type=int, default=0, help='CUDA device index')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=4)

    # 解码器与深度
    parser.add_argument('--decoder', type=str, default='hybrid',
                        choices=['vit', 'mamba', 'hybrid'],
                        help='decoder type: pure ViT / pure Mamba / hybrid (Mamba↔ViT)')
    parser.add_argument('--decoder_attn', type=str, default='linear', choices=['linear', 'full'],
                        help='attention type for ViT blocks in decoder')
    parser.add_argument('--hybrid_pat', type=str, default='v,m,v,m,v,m,v,m',
                        help='pattern for hybrid across N blocks, m= Mamba, v= ViT')
    parser.add_argument('--decode_depth', type=int, default=8,
                        help='number of decoder blocks (default 8 to match your original)')

    # 编码器选择
    parser.add_argument('--encoder', type=str, default='dinov2reg_vit_base_14',
                        help='{dinov2reg_vit_small_14 | dinov2reg_vit_base_14 | dinov2reg_vit_large_14 | ...}')
    parser.add_argument("--bottleneck", type=str, default="geglu_dw",
                        choices=["none","bmlp","geglu_dw","gmlp_dw"])
    parser.add_argument("--bn_mlp_ratio", type=float, default=4.0)
    parser.add_argument("--bn_dw_kernel", type=int, default=5)
    parser.add_argument("--bn_drop", type=float, default=0.1)

    # 优化器 & 学习率调度
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--lr_end', type=float, default=2e-4)
    parser.add_argument('--warmup', type=int, default=100)

    # 日志与评测频率
    parser.add_argument('--eval_every', type=int, default=5000)
    parser.add_argument('--log_every',  type=int, default=100)

    # ===== Step-1 / Step-2 开关与超参（仅评测期生效） =====
    parser.add_argument('--hotspot_gem', action='store_true',
                        help='use Hotspot-GeM aggregator for image-level scores')
    parser.add_argument('--topk_ratio', type=float, default=0.02,
                        help='top-k ratio for hotspot selection, e.g., 0.02=2%')
    parser.add_argument('--gem_p', type=float, default=6.0,
                        help='GeM pooling p for image-level aggregation')
    parser.add_argument('--fg_q', type=int, default=70,
                        help='foreground gate quantile on token norm (0-100)')
    parser.add_argument('--flip_tta', action='store_true',
                        help='use horizontal flip TTA at eval time')
    parser.add_argument('--z_norm', action='store_true',
                        help='class-wise z-normalization for image scores at eval')
    parser.add_argument('--postproc', action='store_true',
                        help='morphological smoothing on heatmap at eval (Step-2)')
    parser.add_argument('--post_k', type=int, default=3, help='morph kernel size (odd)')
    parser.add_argument('--post_iters', type=int, default=1, help='morph iterations')
    parser.add_argument('--rot_tta', type=str, default='textures',
                        help="Rotation TTA policy: 'none'|'all'|'textures' or comma-separated class list")
    parser.add_argument('--ms_tta', type=str, default='0.75,1.0',
                        help="Multi-scale TTA factors, e.g. '0.75,1.0' (1.0 is implicit and will be skipped)")
    parser.add_argument('--tex_rot_tta', action='store_true',
                        help='Enable rotation TTA only for texture classes (carpet/grid/leather/tile/wood).')
    parser.add_argument('--i_agg', type=str, default='max', choices=['max', 'max_topk'],
                        help='Image-level aggregator: max (default) or max_topk')
    parser.add_argument('--i_alpha', type=float, default=0.6,
                        help='alpha for max_topk aggregator (0..1)')
    parser.add_argument('--i_top_percent', type=float, default=5.0,
                        help='top-percent for max_topk aggregator (e.g. 5 means top-5%% pixels)')
    
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


    args = parser.parse_args()

    # 类别列表（VisA 12 类）
    item_list = [
        'candle', 'capsules', 'cashew', 'chewinggum',
        'fryum', 'macaroni1', 'macaroni2',
        'pcb1', 'pcb2', 'pcb3', 'pcb4', 'pipe_fryum'
    ]

    # Logger
    logger   = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info

    # 设备（按命令行选择）
    device = f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu'
    print_fn(device)

    # 训练
    train(item_list)
