#!/usr/bin/env python3
"""
G2L-hybrid t-SNE comparison:
- Panel 1: encoder-only (DINO features)
- Panel 2: ViT-only decoder
- Panel 3: G2L hybrid decoder (e.g., vvvvmmmm)

Markers: support=large circles, normal query=small circles, abnormal query (masked)='x'.
Colors: normal=green, abnormal=red (label mode).
"""
import argparse
import os
import random
from functools import partial
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
from matplotlib import font_manager

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import MVTecDataset, get_data_transforms
from g2l_modules import encoder
from g2l_modules.g2l_model import DALRBlock, G2LHyD
from g2l_modules.gfsr_blocks import GFSRBlock, LinearAttention, SAB


# ---------- Model bits ----------
class ResidualScale(nn.Module):
    def __init__(self, module: nn.Module, init_scale: float = 0.2):
        super().__init__()
        self.module = module
        self.gamma = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))

    def forward(self, x, **kwargs):
        y = self.module(x, **kwargs)
        return x + self.gamma * (y - x)


class PostNorm(nn.Module):
    def __init__(self, module: nn.Module, dim: int, eps: float = 1e-6):
        super().__init__()
        self.module = module
        self.ln = nn.LayerNorm(dim, eps=eps)

    def forward(self, x, **kwargs):
        y = self.module(x, **kwargs)
        return self.ln(y)


def make_decoder(kind: str, dim: int, heads: int, pattern: str, depth: int, num_reg_tokens: int):
    blocks = []
    if kind == "vit":
        for _ in range(depth):
            blocks.append(
                GFSRBlock(
                    dim=dim,
                    num_heads=heads,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    norm_layer=partial(torch.nn.LayerNorm, eps=1e-8),
                    attn=LinearAttention,
                )
            )
    elif kind == "mamba":
        for _ in range(depth):
            blk = DALRBlock(embed_dim=dim, num_register_tokens=num_reg_tokens)
            blk = ResidualScale(blk, init_scale=0.2)
            blk = PostNorm(blk, dim)
            blocks.append(blk)
    else:  # hybrid
        pat = [s.strip().lower() for s in pattern.split(",")]
        if len(pat) != depth:
            raise ValueError("Hybrid pattern length must equal decode_depth")
        for p in pat:
            if p == "v":
                blk = GFSRBlock(
                    dim=dim,
                    num_heads=heads,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    norm_layer=partial(torch.nn.LayerNorm, eps=1e-8),
                    attn=LinearAttention,
                )
            elif p == "m":
                blk = DALRBlock(embed_dim=dim, num_register_tokens=num_reg_tokens)
                blk = ResidualScale(blk, init_scale=0.2)
                blk = PostNorm(blk, dim)
            else:
                raise ValueError(f"Unknown pattern marker {p}")
            blocks.append(blk)
    return torch.nn.ModuleList(blocks)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_fonts(font_path: str, font_name: str):
    if font_path and os.path.exists(font_path):
        try:
            font_manager.fontManager.addfont(font_path)
        except Exception as e:
            print(f"[font] addfont failed for {font_path}: {e}")
    target = font_name or "Times New Roman"
    available = {f.name for f in font_manager.fontManager.ttflist}
    if target in available:
        matplotlib.rcParams["font.family"] = "serif"
        matplotlib.rcParams["font.serif"] = [target]
    else:
        fallback = "DejaVu Serif"
        print(f"[font] {target} not found; fallback to {fallback}.")
        matplotlib.rcParams["font.family"] = "serif"
        matplotlib.rcParams["font.serif"] = [fallback]
    matplotlib.rcParams["mathtext.fontset"] = "stix"
    matplotlib.rcParams["axes.unicode_minus"] = False


def build_model(args, decoder_kind: str, pattern: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder = encoder.load(args.encoder)
    if "small" in args.encoder:
        embed_dim, num_heads = 384, 6
    elif "base" in args.encoder:
        embed_dim, num_heads = 768, 12
    elif "large" in args.encoder:
        embed_dim, num_heads = 1024, 16
    else:
        raise RuntimeError("encoder must contain small/base/large.")

    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]

    bottleneck = torch.nn.ModuleList(
        [SAB(embed_dim, hidden_features=embed_dim * 2, out_features=embed_dim, drop=0.1, grad=1.0, dw_kernel=5)]
    )
    decoder = make_decoder(decoder_kind, embed_dim, num_heads, pattern, args.decode_depth, args.num_reg_tokens)

    model = G2LHyD(
        encoder=encoder,
        bottleneck=bottleneck,
        decoder=decoder,
        target_layers=target_layers,
        fuse_layer_encoder=fuse_layer_encoder,
        fuse_layer_decoder=fuse_layer_decoder,
    )

    if args.model_path:
        try:
            state = torch.load(args.model_path, map_location=device, weights_only=False)
        except TypeError:
            state = torch.load(args.model_path, map_location=device)
        raw_state = state.get("model", state)
        model_state = model.state_dict()
        filtered = {}
        skipped = []
        for k, v in raw_state.items():
            if k not in model_state or model_state[k].shape != v.shape:
                skipped.append(k)
                continue
            filtered[k] = v
        missing, unexpected = model.load_state_dict(filtered, strict=False)
        if skipped:
            print(f"[{decoder_kind}] skipped {len(skipped)} keys (e.g., {skipped[:3]})")
        if missing:
            print(f"[{decoder_kind}] missing {len(missing)} keys (e.g., {missing[:3]})")
        if unexpected:
            print(f"[{decoder_kind}] unexpected {len(unexpected)} keys (e.g., {unexpected[:3]})")

    model.to(device)
    model.eval()
    return model, device


def default_root():
    return Path("data/mvtec_anomaly_detection")


def get_support_loader(root: Path, cls: str, transform, gt_transform, k: int, batch_size: int):
    ds = MVTecDataset(root=str(root / cls), transform=transform, gt_transform=gt_transform, phase="train")
    if k > 0:
        ds.img_paths = ds.img_paths[:k]
        ds.gt_paths = ds.gt_paths[:k]
        ds.labels = ds.labels[:k]
        ds.types = ds.types[:k]
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


def get_query_loader(root: Path, cls: str, transform, gt_transform, batch_size: int):
    ds = MVTecDataset(root=str(root / cls), transform=transform, gt_transform=gt_transform, phase="test")
    return DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0)


def pick_feat(kind: str, en_list, de_list):
    if kind == "encoder":
        return en_list[-1]
    return de_list[-1]


def gather_features(
    model,
    loader_support,
    loader_query,
    device,
    kind: str,
    max_normal_query: int,
    max_abnormal: int,
    max_normal_from_abnormal: int,
):
    feats = []
    labels = []  # 0=normal,1=abnormal
    roles = []  # support/normal/abnormal

    with torch.no_grad():
        # support
        for imgs, _, _, _ in loader_support:
            imgs = imgs.to(device)
            en, de = model(imgs)
            fmap = pick_feat(kind, en, de)
            fmap = F.normalize(fmap, dim=1)
            fh, fw = fmap.shape[-2], fmap.shape[-1]
            pts = fmap.permute(0, 2, 3, 1).reshape(-1, fmap.shape[1])
            feats.append(pts.cpu())
            labels.extend([0] * pts.shape[0])
            roles.extend(["support"] * pts.shape[0])

        normal_kept = 0
        abnormal_kept = 0
        for imgs, gt, lbls, _ in tqdm(loader_query, desc=f"{kind} query", leave=False):
            imgs = imgs.to(device)
            gt = gt.to(device)
            lbls = lbls.to(device)
            en, de = model(imgs)
            fmap = pick_feat(kind, en, de)
            fmap = F.normalize(fmap, dim=1)
            fh, fw = fmap.shape[-2], fmap.shape[-1]
            pts = fmap.permute(0, 2, 3, 1).reshape(imgs.shape[0], fh * fw, fmap.shape[1])
            # mask resize to patch grid
            gt_grid = F.interpolate(gt, size=(fh, fw), mode="nearest").flatten(2)  # B x 1 x HW

            for i in range(imgs.shape[0]):
                patches = pts[i]
                mask = gt_grid[i, 0] > 0.5
                if lbls[i] == 0:  # normal query
                    if normal_kept >= max_normal_query:
                        continue
                    idxs = torch.randperm(patches.shape[0], device=patches.device)
                    take = min(max_normal_query - normal_kept, patches.shape[0])
                    keep = idxs[:take]
                    feats.append(patches[keep].cpu())
                    labels.extend([0] * take)
                    roles.extend(["normal"] * take)
                    normal_kept += take
                else:  # abnormal image
                    # abnormal patches: mask==1
                    ab_idx = torch.nonzero(mask, as_tuple=False).squeeze(-1)
                    if ab_idx.numel() > 0 and abnormal_kept < max_abnormal:
                        take = ab_idx
                        if abnormal_kept + take.numel() > max_abnormal:
                            take = take[: max_abnormal - abnormal_kept]
                        feats.append(patches[take].cpu())
                        labels.extend([1] * take.numel())
                        roles.extend(["abnormal"] * take.numel())
                        abnormal_kept += take.numel()
                    # normal patches from abnormal image (background)
                    if max_normal_from_abnormal > 0:
                        bg_idx = torch.nonzero(~mask, as_tuple=False).squeeze(-1)
                        if bg_idx.numel() > 0:
                            take = min(max_normal_from_abnormal, bg_idx.numel())
                            sel = bg_idx[torch.randperm(bg_idx.numel(), device=bg_idx.device)[:take]]
                            feats.append(patches[sel].cpu())
                            labels.extend([0] * take)
                            roles.extend(["normal"] * take)

    if feats:
        feats_cat = torch.cat(feats, dim=0)
    else:
        feats_cat = torch.empty((0, 1))
    return feats_cat.numpy(), np.array(labels), np.array(roles)


def run_tsne(feats: np.ndarray, args):
    if feats.shape[0] < 2:
        return np.zeros((feats.shape[0], 2))
    feats_proc = feats
    if args.pca_dim and feats.shape[1] > args.pca_dim:
        feats_proc = PCA(n_components=args.pca_dim, random_state=args.seed).fit_transform(feats_proc)
    perplexity = min(args.perplexity, max(5, feats_proc.shape[0] - 1))
    tsne = TSNE(
        n_components=2,
        perplexity=perplexity,
        n_iter=args.tsne_iter,
        early_exaggeration=args.early_exaggeration,
        learning_rate=args.lr,
        metric=args.metric,
        init="pca",
        random_state=args.seed,
    )
    return tsne.fit_transform(feats_proc)


def plot_panels(coords_list, labels, roles, args, out_path: Path):
    # Colors by label (normal/abnormal)
    palette = {"normal": args.normal_color or "#1f77b4", "abnormal": args.anomaly_color or "#d62728"}
    markers = {"support": "o", "normal": "o", "abnormal": "x"}
    sizes = {"support": args.point_size * 1.8, "normal": args.point_size * 0.8, "abnormal": args.point_size * 1.2}
    alphas = {"support": 0.7, "normal": 0.5, "abnormal": 1.0}

    titles = ["Encoder only", "ViT decoder (global)", "G2L hybrid"]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for idx_panel, ax in enumerate(axes):
        coords = coords_list[idx_panel]
        for role in ["support", "normal", "abnormal"]:
            idxs = np.where(roles == role)[0]
            if idxs.size == 0:
                continue
            lbls = labels[idxs]
            colors = [palette["abnormal" if l == 1 else "normal"] for l in lbls]
            ax.scatter(
                coords[idxs, 0],
                coords[idxs, 1],
                s=sizes[role],
                c=colors,
                alpha=alphas[role],
                linewidths=0.5 if role == "abnormal" else 0.0,
                edgecolors="white" if role == "abnormal" else "none",
                marker=markers[role],
                zorder=2 if role != "abnormal" else 3,
            )
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(titles[idx_panel])

    # legend
    from matplotlib.lines import Line2D

    legend_elems = [
        Line2D([0], [0], marker="o", color="w", label="Support (normal)", markerfacecolor=palette["normal"], markersize=8),
        Line2D([0], [0], marker="o", color="w", label="Query normal", markerfacecolor=palette["normal"], markersize=6, alpha=0.6),
        Line2D([0], [0], marker="x", color=palette["abnormal"], label="Query abnormal", markersize=7, markeredgewidth=1.5),
    ]
    axes[0].legend(handles=legend_elems, loc="upper right", frameon=False, fontsize=9)

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved figure to {out_path}")


def parse_args():
    p = argparse.ArgumentParser("G2L-hybrid t-SNE (encoder vs ViT vs hybrid)")
    p.add_argument("--dataset_root", type=str, default=None, help="Root for MVTec.")
    p.add_argument("--class_name", type=str, default="screw", help="Single class to visualize.")
    p.add_argument("--model_path", type=str, required=True)
    p.add_argument("--encoder", type=str, default="dinov2reg_vit_base_14")
    p.add_argument("--decode_depth", type=int, default=8)
    p.add_argument("--num_reg_tokens", type=int, default=4)
    p.add_argument("--hybrid_pat", type=str, default="v,v,v,v,m,m,m,m", help="Pattern for hybrid decoder.")
    p.add_argument("--image_size", type=int, default=448)
    p.add_argument("--crop_size", type=int, default=392)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--support_k", type=int, default=1, help="Number of support images (train/good).")
    p.add_argument("--max_normal_query", type=int, default=300)
    p.add_argument("--max_abnormal", type=int, default=600)
    p.add_argument("--max_normal_from_abnormal", type=int, default=120)
    p.add_argument("--pca_dim", type=int, default=50)
    p.add_argument("--perplexity", type=float, default=35.0)
    p.add_argument("--early_exaggeration", type=float, default=16.0)
    p.add_argument("--tsne_iter", type=int, default=1200)
    p.add_argument("--lr", type=float, default=400.0)
    p.add_argument("--metric", choices=["euclidean", "cosine"], default="euclidean")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--point_size", type=float, default=10.0)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--output", type=str, default="data/Visual/tsne_g2l.png")
    p.add_argument("--normal_color", type=str, default="#2ca02c")
    p.add_argument("--anomaly_color", type=str, default="#d62728")
    p.add_argument("--font_path", type=str, default="", help="TTF path for Times New Roman if needed.")
    p.add_argument("--font_name", type=str, default="Times New Roman")
    return p.parse_args()


def main():
    args = parse_args()
    configure_fonts(args.font_path, args.font_name)
    set_seed(args.seed)
    root = Path(args.dataset_root) if args.dataset_root else default_root()

    data_transform, gt_transform = get_data_transforms(args.image_size, args.crop_size)
    support_loader = get_support_loader(root, args.class_name, data_transform, gt_transform, args.support_k, args.batch_size)
    query_loader = get_query_loader(root, args.class_name, data_transform, gt_transform, args.batch_size)

    coords_list = []
    labels_ref = None
    roles_ref = None

    variants = [
        ("encoder", "hybrid", args.hybrid_pat),  # use encoder features only
        ("vit", "vit", ""),  # ViT-only decoder
        ("hybrid", "hybrid", args.hybrid_pat),  # G2L hybrid decoder
    ]

    for kind, decoder_kind, pattern in variants:
        # ensure identical sampling across panels
        set_seed(args.seed)
        model, device = build_model(args, decoder_kind if kind != "encoder" else "hybrid", pattern or args.hybrid_pat)
        feats, lbls, roles = gather_features(
            model,
            support_loader,
            query_loader,
            device,
            "encoder" if kind == "encoder" else decoder_kind,
            args.max_normal_query,
            args.max_abnormal,
            args.max_normal_from_abnormal,
        )
        coords = run_tsne(feats, args)
        coords_list.append(coords)
        if labels_ref is None:
            labels_ref = lbls
            roles_ref = roles

    out_path = Path(args.output)
    plot_panels(coords_list, labels_ref, roles_ref, args, out_path)


if __name__ == "__main__":
    main()
