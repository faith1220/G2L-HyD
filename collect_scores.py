#!/usr/bin/env python3
"""
collect_scores.py  —  Reviewer实验第1步：推理收集原始异常分数

用途：
  对指定 decoder 变体的 checkpoint 进行推理，收集每个类别每张测试图的
  image-level anomaly score 和 label，保存为 .npz 文件。

输出文件 (每次运行一个变体):
  <output_dir>/<variant_name>.npz
  内容:
    scores[category]  = np.array  (每张图的 image-level anomaly score)
    labels[category]  = np.array  (0=normal, 1=anomalous)
    pixel_scores[category] = list of np.array  (每张图的 pixel-level anomaly map, 可选)

用法:
  # G2L (论文默认)
  python collect_scores.py \
    --data_path /path/to/mvtec_anomaly_detection/ \
    --load_ckpt /path/to/g2l_ckpt.pth \
    --variant_name G2L \
    --hybrid_pat v,v,v,v,m,m,m,m \
    --dataset mvtec

  # L2G
  python collect_scores.py \
    --data_path /path/to/mvtec_anomaly_detection/ \
    --load_ckpt /path/to/l2g_ckpt.pth \
    --variant_name L2G \
    --hybrid_pat m,m,m,m,v,v,v,v \
    --dataset mvtec

  # All-DALR
  python collect_scores.py \
    --data_path /path/to/mvtec_anomaly_detection/ \
    --load_ckpt /path/to/all_dalr_ckpt.pth \
    --variant_name All_DALR \
    --decoder mamba \
    --dataset mvtec

  # All-GFSR
  python collect_scores.py \
    --data_path /path/to/mvtec_anomaly_detection/ \
    --load_ckpt /path/to/all_gfsr_ckpt.pth \
    --variant_name All_GFSR \
    --decoder vit \
    --dataset mvtec
"""

import os
import sys
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import argparse
import math
import functools
import json

from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets import ImageFolder

# ====== 项目内模块 ======
from dataset import get_data_transforms, MVTecDataset, RealIADDataset
from g2l_modules.g2l_model import G2LHyD, DALRBlock
from g2l_modules import encoder
from g2l_modules.gfsr_blocks import (
    GFSRBlock, Attention, LinearAttention,
    SAB, LinearGMLPTokenBlock
)
from utils import cal_anomaly_maps


# ======================== 工具函数 ========================
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _strip_module_prefix(state_dict):
    return {(k[7:] if k.startswith("module.") else k): v for k, v in state_dict.items()}


def _extract_state_dict(ckpt_obj):
    if isinstance(ckpt_obj, dict):
        if "model" in ckpt_obj and isinstance(ckpt_obj["model"], dict):
            return ckpt_obj["model"]
        if "state_dict" in ckpt_obj and isinstance(ckpt_obj["state_dict"], dict):
            return ckpt_obj["state_dict"]
    return ckpt_obj


def _load_state_dict_with_fallback(model, state_dict):
    state_dict = _strip_module_prefix(state_dict)
    model_state = model.state_dict()
    matched = {}
    for k, v in state_dict.items():
        if k in model_state and model_state[k].shape == v.shape:
            matched[k] = v
    model.load_state_dict(matched, strict=False)
    return len(matched), len(state_dict) - len(matched)


class ResidualScale(nn.Module):
    def __init__(self, module, init_scale=0.2):
        super().__init__()
        self.module = module
        self.gamma = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))

    def forward(self, x, **kwargs):
        y = self.module(x, **kwargs)
        return x + self.gamma * (y - x)


class PostNorm(nn.Module):
    def __init__(self, module, dim, eps=1e-6):
        super().__init__()
        self.module = module
        self.ln = nn.LayerNorm(dim, eps=eps)

    def forward(self, x, **kwargs):
        y = self.module(x, **kwargs)
        return self.ln(y)


# ======================== 模型构建 ========================
def make_vit_block(dim, heads, attn_type='linear'):
    attn_class = Attention if attn_type == 'full' else LinearAttention
    return GFSRBlock(
        dim=dim, num_heads=heads, mlp_ratio=4., qkv_bias=True,
        norm_layer=functools.partial(nn.LayerNorm, eps=1e-8),
        attn=attn_class
    )


def make_decoder(kind, dim, heads, pattern=None, depth=8, num_reg_tokens=4, args=None):
    blocks = []

    def make_mamba_block():
        ks = tuple(int(x) for x in args.mamba_ks.split(','))
        dils = tuple(int(x) for x in args.mamba_dilations.split(','))
        if args.mamba_dw_kernel > 0:
            ks = (args.mamba_dw_kernel, args.mamba_dw_kernel)
        lss_kwargs = dict(
            ks=ks, dilations=dils,
            add_local3=bool(args.mamba_add_local3),
            add_asym=bool(args.mamba_add_asym),
            hss_learn_dir=True
        )
        blk = DALRBlock(
            embed_dim=dim, num_register_tokens=num_reg_tokens,
            num_hss=int(args.mamba_num_hss),
            scan_method=args.mamba_scan_method,
            num_scan_dirs=args.mamba_scan_dirs,
            lss_kwargs=lss_kwargs
        )
        blk = ResidualScale(blk, init_scale=0.2)
        blk = PostNorm(blk, dim)
        return blk

    if kind == 'vit':
        for _ in range(depth):
            blocks.append(make_vit_block(dim, heads, args.decoder_attn))
    elif kind == 'mamba':
        for _ in range(depth):
            blocks.append(make_mamba_block())
    else:  # hybrid
        pat = [s.strip().lower() for s in (pattern or 'v,m,v,m,v,m,v,m').split(',')]
        assert len(pat) == depth
        for p in pat:
            if p == 'm':
                blocks.append(make_mamba_block())
            elif p == 'v':
                blocks.append(make_vit_block(dim, heads, args.decoder_attn))
            else:
                raise ValueError(f"Unknown token: {p}")
    return nn.ModuleList(blocks)


def make_bottleneck(embed_dim, variant, args, num_reg_tokens=4):
    if variant == 'none':
        return nn.Identity()
    elif variant == 'gmlp_dw':
        return SAB(
            in_features=embed_dim, hidden_features=int(2 * embed_dim),
            out_features=embed_dim,
            drop=args.bn_drop, grad=0.7, dw_kernel=args.bn_dw_kernel,
            num_register_tokens=num_reg_tokens, has_cls=True
        )
    elif variant == 'gmlp_lin':
        return LinearGMLPTokenBlock(
            in_features=embed_dim, hidden_features=int(2 * embed_dim),
            out_features=embed_dim,
            drop=args.bn_drop, grad=0.7,
            num_register_tokens=num_reg_tokens, has_cls=True,
            token_mode='banded', bandwidth=7, rank=64
        )
    else:
        raise ValueError(f"Unsupported bottleneck variant for score collection: {variant}")


def build_model(args, device):
    encoder = encoder.load(args.encoder)
    num_reg_tokens = 4

    if 'small' in args.encoder:
        embed_dim, num_heads = 384, 6
    elif 'base' in args.encoder:
        embed_dim, num_heads = 768, 12
    elif 'large' in args.encoder:
        embed_dim, num_heads = 1024, 16
    else:
        raise RuntimeError("Encoder not in small/base/large.")

    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

    bottleneck_module = make_bottleneck(embed_dim, args.bottleneck_variant, args, num_reg_tokens)
    bottleneck = nn.ModuleList([bottleneck_module])

    decoder = make_decoder(
        args.decoder, embed_dim, num_heads,
        pattern=args.hybrid_pat, depth=args.decode_depth,
        num_reg_tokens=num_reg_tokens, args=args
    )

    model = G2LHyD(
        encoder=encoder, bottleneck=bottleneck, decoder=decoder,
        target_layers=target_layers,
        fuse_layer_encoder=fuse_layer_encoder,
        fuse_layer_decoder=fuse_layer_decoder
    )
    model = model.to(device)

    # 加载 checkpoint
    if args.load_ckpt:
        ckpt_path = os.path.expanduser(args.load_ckpt)
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = _extract_state_dict(state)
        matched, unmatched = _load_state_dict_with_fallback(model, state_dict)
        print(f"[ckpt] loaded from {ckpt_path} | matched={matched} | unmatched={unmatched}")

    model.eval()
    return model


# ======================== 数据集加载 ========================
MVTEC_CLASSES = [
    'carpet', 'grid', 'leather', 'tile', 'wood',
    'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut',
    'pill', 'screw', 'toothbrush', 'transistor', 'zipper'
]

VISA_CLASSES = [
    'candle', 'capsules', 'cashew', 'chewinggum', 'fryum',
    'macaroni1', 'macaroni2', 'pcb1', 'pcb2', 'pcb3', 'pcb4',
    'pipe_fryum'
]


def load_test_datasets(args):
    """返回 (item_list, test_data_list)"""
    image_size, crop_size = 448, 392
    data_transform, gt_transform = get_data_transforms(image_size, crop_size)

    if args.dataset == 'mvtec':
        item_list = MVTEC_CLASSES
        test_data_list = []
        for item in item_list:
            test_path = os.path.join(args.data_path, item)
            test_data = MVTecDataset(root=test_path, transform=data_transform,
                                     gt_transform=gt_transform, phase="test")
            test_data_list.append(test_data)

    elif args.dataset == 'visa':
        item_list = VISA_CLASSES
        test_data_list = []
        for item in item_list:
            test_path = os.path.join(args.data_path, item)
            test_data = MVTecDataset(root=test_path, transform=data_transform,
                                     gt_transform=gt_transform, phase="test")
            test_data_list.append(test_data)

    elif args.dataset == 'realiad':
        # Real-IAD 使用不同的 Dataset 类
        item_list = sorted(os.listdir(os.path.join(args.data_path, 'realiad_1024')))
        test_data_list = []
        for item in item_list:
            test_data = RealIADDataset(
                root=args.data_path, category=item,
                transform=data_transform, gt_transform=gt_transform, phase="test"
            )
            test_data_list.append(test_data)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    return item_list, test_data_list


# ======================== 分数收集核心 ========================
@torch.no_grad()
def collect_scores_for_variant(model, item_list, test_data_list, device, args):
    """
    对每个类别跑推理，收集 image-level anomaly score 和 label。
    返回: {category: {"scores": np.array, "labels": np.array}}
    """
    model.eval()
    results = {}

    for item, test_data in zip(item_list, test_data_list):
        test_loader = DataLoader(
            test_data, batch_size=args.batch_size,
            shuffle=False, num_workers=args.workers
        )

        scores_list = []
        labels_list = []

        for img, gt, label, img_path in test_loader:
            img = img.to(device)

            en, de = model(img)
            anomaly_map, _ = cal_anomaly_maps(en, de, img.shape[-1])

            # image-level score: top-k% 的均值
            B = anomaly_map.shape[0]
            amap_flat = anomaly_map.flatten(1)  # [B, H*W]
            k = max(1, int(amap_flat.shape[1] * 0.01))
            sp_score = torch.sort(amap_flat, dim=1, descending=True)[0][:, :k].mean(dim=1)

            scores_list.append(sp_score.cpu().numpy())
            labels_list.append(label.numpy() if isinstance(label, torch.Tensor) else np.array(label))

        scores_arr = np.concatenate(scores_list)
        labels_arr = np.concatenate(labels_list)

        results[item] = {
            "scores": scores_arr,
            "labels": labels_arr
        }
        print(f"  [{item}] n_normal={np.sum(labels_arr == 0)}, "
              f"n_anomaly={np.sum(labels_arr == 1)}, "
              f"score_range=[{scores_arr.min():.4f}, {scores_arr.max():.4f}]")

    return results


# ======================== 主函数 ========================
def main():
    parser = argparse.ArgumentParser(description='Collect anomaly scores for reviewer experiments')
    # 数据
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--dataset', type=str, default='mvtec', choices=['mvtec', 'visa', 'realiad'])
    # 模型
    parser.add_argument('--load_ckpt', type=str, required=True)
    parser.add_argument('--encoder', type=str, default='dinov2reg_vit_base_14')
    parser.add_argument('--decoder', type=str, default='hybrid', choices=['vit', 'mamba', 'hybrid'])
    parser.add_argument('--decoder_attn', type=str, default='linear', choices=['linear', 'full'])
    parser.add_argument('--hybrid_pat', type=str, default='v,v,v,v,m,m,m,m')
    parser.add_argument('--decode_depth', type=int, default=8)
    parser.add_argument('--bottleneck_variant', type=str, default='gmlp_dw')
    parser.add_argument('--bn_drop', type=float, default=0.1)
    parser.add_argument('--bn_dw_kernel', type=int, default=5)
    # Mamba 参数
    parser.add_argument('--mamba_ks', type=str, default='5,7')
    parser.add_argument('--mamba_dilations', type=str, default='1,2')
    parser.add_argument('--mamba_dw_kernel', type=int, default=0)
    parser.add_argument('--mamba_add_local3', type=int, default=1)
    parser.add_argument('--mamba_add_asym', type=int, default=1)
    parser.add_argument('--mamba_num_hss', type=int, default=3)
    parser.add_argument('--mamba_scan_method', type=str, default='hilbert')
    parser.add_argument('--mamba_scan_dirs', type=int, default=8)
    # 运行
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--cuda', type=int, default=0)
    parser.add_argument('--seed', type=int, default=1)
    # 输出
    parser.add_argument('--variant_name', type=str, required=True,
                        help='Name for this decoder variant, e.g., G2L, L2G, All_DALR, All_GFSR')
    parser.add_argument('--output_dir', type=str, default='./reviewer_analysis/scores')

    args = parser.parse_args()

    setup_seed(args.seed)
    device = f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"=== Collecting scores for variant: {args.variant_name} ===")
    print(f"  Dataset: {args.dataset}")
    print(f"  Decoder: {args.decoder}, Pattern: {args.hybrid_pat}")
    print(f"  Checkpoint: {args.load_ckpt}")

    # 构建模型
    model = build_model(args, device)

    # 加载测试数据
    item_list, test_data_list = load_test_datasets(args)

    # 收集分数
    results = collect_scores_for_variant(model, item_list, test_data_list, device, args)

    # 保存
    save_dict = {}
    for cat, data in results.items():
        save_dict[f"scores_{cat}"] = data["scores"]
        save_dict[f"labels_{cat}"] = data["labels"]
    save_dict["categories"] = np.array(list(results.keys()))

    save_path = os.path.join(args.output_dir, f"{args.variant_name}.npz")
    np.savez(save_path, **save_dict)
    print(f"\nSaved scores to: {save_path}")


if __name__ == '__main__':
    main()
