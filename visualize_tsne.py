#!/usr/bin/env python3
import argparse
import io
import json
import os
import random
import re
import zipfile
from functools import partial
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import matplotlib
from matplotlib import font_manager

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from scipy.spatial import ConvexHull
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from dataset import MVTecDataset, get_data_transforms
from g2l_modules import encoder
from g2l_modules.g2l_model import DALRBlock, G2LHyD
from g2l_modules.gfsr_blocks import GFSRBlock, LinearAttention, SAB


# --------- Model helpers ---------
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


def make_decoder(kind: str, dim: int, heads: int, pattern: str = None, depth: int = 8, num_reg_tokens: int = 4):
    blocks = []
    if kind == "vit":
        for _ in range(depth):
            blocks.append(
                GFSRBlock(
                    dim=dim,
                    num_heads=heads,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    norm_layer=partial(torch.nn.LayerNorm, eps=1e-8),  # type: ignore[name-defined]
                    attn=LinearAttention,
                )
            )
    elif kind == "mamba":
        for _ in range(depth):
            blk = DALRBlock(embed_dim=dim, num_register_tokens=num_reg_tokens)
            blk = ResidualScale(blk, init_scale=0.2)
            blk = PostNorm(blk, dim)
            blocks.append(blk)
    else:  # hybrid decoder
        pat = [s.strip().lower() for s in (pattern or "v,m,v,m,v,m,v,m").split(",")]
        if len(pat) != depth:
            raise ValueError("Hybrid pattern length must match decode_depth.")
        for p in pat:
            if p == "m":
                blk = DALRBlock(embed_dim=dim, num_register_tokens=num_reg_tokens)
                blk = ResidualScale(blk, init_scale=0.2)
                blk = PostNorm(blk, dim)
            elif p == "v":
                blk = GFSRBlock(
                    dim=dim,
                    num_heads=heads,
                    mlp_ratio=4.0,
                    qkv_bias=True,
                    norm_layer=partial(torch.nn.LayerNorm, eps=1e-8),  # type: ignore[name-defined]
                    attn=LinearAttention,
                )
            else:
                raise ValueError(f"Unknown marker: {p}")
            blocks.append(blk)
    return torch.nn.ModuleList(blocks)


# --------- Real-IAD dataset helper ---------
def load_realiad_json(root: Path, category: str) -> Dict:
    json_dir = root / "realiad_jsons" / "realiad_jsons"
    json_file = json_dir / f"{category}.json"
    if json_file.exists():
        with open(json_file, "r", encoding="utf-8") as f:
            return json.load(f)
    zip_path = root / "realiad_jsons.zip"
    if zip_path.exists():
        with zipfile.ZipFile(zip_path) as zf:
            target = f"realiad_jsons/{category}.json"
            if target not in zf.namelist():
                raise FileNotFoundError(f"{target} not found inside {zip_path}.")
            with zf.open(target) as fp:
                return json.load(io.TextIOWrapper(fp, encoding="utf-8"))
    raise FileNotFoundError(f"Missing Real-IAD json for {category} under {root}.")


def parse_view_tag(rel_path: str) -> str:
    m = re.search(r"_C(\\d+)_", rel_path)
    if m:
        return f"C{m.group(1)}"
    return "view?"


def parse_object_id(rel_path: str) -> str:
    m = re.search(r"/(S\\d+)/", rel_path)
    if m:
        return m.group(1)
    m = re.search(r"(S\\d+)", rel_path)
    if m:
        return m.group(1)
    stem = os.path.basename(rel_path)
    return stem.split("_")[0]


class RealIADTSNEDataset(Dataset):
    def __init__(self, root: Path, category: str, transform, phase: str = "test", resolution: int = 1024):
        meta = load_realiad_json(root, category)
        data = meta.get(phase)
        if data is None:
            raise ValueError(f"Phase {phase} missing in json for {category}.")
        self.samples = []
        for sample in data:
            rel_img = sample["image_path"]
            img_path = root / f"realiad_{resolution}" / category / rel_img
            if not img_path.exists():
                continue
            label = 0 if sample["anomaly_class"] == "OK" else 1
            view = parse_view_tag(rel_img)
            obj_id = parse_object_id(rel_img)
            self.samples.append((str(img_path), label, view, obj_id))
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, view, obj_id = self.samples[idx]
        img = Image.open(path).convert("RGB")
        img = self.transform(img)
        return img, label, path, view, obj_id


# --------- Utility ---------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_fonts(font_path: str, font_name: str):
    if font_path:
        if os.path.exists(font_path):
            try:
                font_manager.fontManager.addfont(font_path)
            except Exception as e:
                print(f"[font] addfont failed for {font_path}: {e}")
        else:
            print(f"[font] font_path {font_path} not found, will try system fonts.")
    available = {f.name for f in font_manager.fontManager.ttflist}
    target = font_name if font_name else "Times New Roman"
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


def default_root(dataset: str) -> Path:
    if dataset == "mvtec":
        return Path("data/mvtec_anomaly_detection")
    if dataset == "visa":
        return Path("data/VisA_pytorch/1cls")
    if dataset == "realiad":
        return Path("data/Real-IAD")
    raise ValueError(f"Unsupported dataset {dataset}.")


def list_classes(dataset: str, root: Path) -> List[str]:
    if dataset in ("mvtec", "visa"):
        if not root.exists():
            raise FileNotFoundError(f"Root {root} not found.")
        return sorted([d for d in os.listdir(root) if (root / d).is_dir()])
    if dataset == "realiad":
        json_dir = root / "realiad_jsons" / "realiad_jsons"
        if json_dir.exists():
            return sorted([p.stem for p in json_dir.glob("*.json")])
        zip_path = root / "realiad_jsons.zip"
        if zip_path.exists():
            with zipfile.ZipFile(zip_path) as zf:
                classes = [Path(n).stem for n in zf.namelist() if n.startswith("realiad_jsons/") and n.endswith(".json")]
            return sorted(classes)
        raise FileNotFoundError(f"Cannot locate Real-IAD metadata under {root}.")
    raise ValueError(f"Unsupported dataset {dataset}.")


def build_model(args) -> Tuple[G2LHyD, torch.device]:
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
    decoder = make_decoder(
        args.decoder,
        embed_dim,
        num_heads,
        pattern=args.hybrid_pat,
        depth=args.decode_depth,
        num_reg_tokens=args.num_reg_tokens,
    )
    model = G2LHyD(
        encoder=encoder,
        bottleneck=bottleneck,
        decoder=decoder,
        target_layers=target_layers,
        fuse_layer_encoder=fuse_layer_encoder,
        fuse_layer_decoder=fuse_layer_decoder,
    )
    if args.model_path:
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f"model_path {args.model_path} not found.")
        try:
            state = torch.load(args.model_path, map_location=device, weights_only=False)
        except TypeError:
            state = torch.load(args.model_path, map_location=device)
        raw_state = state.get("model", state)
        model_state = model.state_dict()
        filtered = {}
        skipped = []
        for k, v in raw_state.items():
            if k not in model_state:
                skipped.append(k)
                continue
            if model_state[k].shape != v.shape:
                skipped.append(k)
                continue
            filtered[k] = v
        missing_keys, unexpected_keys = model.load_state_dict(filtered, strict=False)
        if skipped:
            print(f"[load_state_dict] skipped {len(skipped)} mismatched/extra keys (e.g., {skipped[:3]})")
        if missing_keys:
            print(f"[load_state_dict] missing keys: {missing_keys[:3]} (total {len(missing_keys)})")
        if unexpected_keys:
            print(f"[load_state_dict] unexpected keys: {unexpected_keys[:3]} (total {len(unexpected_keys)})")
    model.to(device)
    model.eval()
    return model, device


def build_loaders(args, data_root: Path, data_transform, gt_transform):
    loaders = []
    selected = args.classes
    all_classes = list_classes(args.dataset, data_root)
    if selected:
        required = [c.strip() for c in selected.split(",") if c.strip()]
        missing = [c for c in required if c not in all_classes]
        if missing:
            raise ValueError(f"Classes {missing} not found under {data_root}.")
        class_list = required
    else:
        class_list = all_classes

    for cls in class_list:
        if args.dataset in ("mvtec", "visa"):
            root = data_root / cls
            if not root.exists():
                continue
            ds = MVTecDataset(root=str(root), transform=data_transform, gt_transform=gt_transform, phase="test")
        elif args.dataset == "realiad":
            ds = RealIADTSNEDataset(
                root=data_root, category=cls, transform=data_transform, phase="test", resolution=args.realiad_resolution
            )
        else:
            raise ValueError(f"Unsupported dataset {args.dataset}.")
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)
        loaders.append((cls, loader))
    return loaders


def unpack_batch(dataset: str, batch):
    if dataset in ("mvtec", "visa"):
        imgs, _, labels, paths = batch
        views = ["C0"] * len(labels)
        obj_ids = [None] * len(labels)
    elif dataset == "realiad":
        imgs, labels, paths, views, obj_ids = batch
    else:
        raise ValueError(f"Unsupported dataset {dataset}.")
    labels = labels.cpu().numpy().tolist()
    paths = [p for p in paths]
    views = [v for v in views]
    return imgs, labels, paths, views, obj_ids


def reduce_and_normalize(feat: torch.Tensor) -> torch.Tensor:
    pooled = feat.mean(dim=[2, 3])  # B x C
    return F.normalize(pooled, dim=1)


def collect_embeddings(model, loaders, device, args):
    enc_list, dec_list, labels_out, class_tags, view_tags, obj_ids = [], [], [], [], [], []
    total_limit = args.max_total if args.max_total > 0 else None
    with torch.no_grad():
        for cls_name, loader in loaders:
            class_counts = {"normal": 0, "anomaly": 0}
            pbar = tqdm(loader, desc=f"Extract {cls_name}", leave=False)
            for batch in pbar:
                imgs, labels, paths, views, objs = unpack_batch(args.dataset, batch)
                mask = []
                for lbl in labels:
                    key = "anomaly" if lbl else "normal"
                    limit = args.max_anomaly_per_class if lbl else args.max_normal_per_class
                    if limit > 0 and class_counts[key] >= limit:
                        mask.append(False)
                    else:
                        class_counts[key] += 1
                        mask.append(True)
                if not any(mask):
                    continue
                mask_tensor = torch.tensor(mask, device=imgs.device, dtype=torch.bool)
                imgs_sel = imgs[mask_tensor].to(device, non_blocking=True)
                labels_sel = [l for l, keep in zip(labels, mask) if keep]
                paths_sel = [p for p, keep in zip(paths, mask) if keep]
                views_sel = [v for v, keep in zip(views, mask) if keep]
                objs_sel = [o for o, keep in zip(objs, mask) if keep]

                en_maps, de_maps = model(imgs_sel)
                enc = reduce_and_normalize(en_maps[-1]).cpu()
                dec = reduce_and_normalize(de_maps[-1]).cpu()
                enc_list.append(enc)
                dec_list.append(dec)
                labels_out.extend(labels_sel)
                class_tags.extend([cls_name] * len(labels_sel))
                view_tags.extend(views_sel)
                obj_ids.extend(objs_sel)

                if total_limit is not None and len(labels_out) >= total_limit:
                    break
            if total_limit is not None and len(labels_out) >= total_limit:
                break

    enc_feats = torch.cat(enc_list, dim=0) if enc_list else torch.empty((0, 1))
    dec_feats = torch.cat(dec_list, dim=0) if dec_list else torch.empty((0, 1))
    return enc_feats, dec_feats, labels_out, class_tags, view_tags, obj_ids


def aggregate_multi_view(enc, dec, labels, classes, views, obj_ids, strategy: str):
    if strategy != "multi_view_mean":
        return enc, dec, labels, classes, views
    grouped: Dict[Tuple[str, int, str], List[int]] = {}
    for idx, (cls, lbl, obj) in enumerate(zip(classes, labels, obj_ids)):
        key = (cls, lbl, obj or f"idx{idx}")
        grouped.setdefault(key, []).append(idx)
    new_enc, new_dec, new_labels, new_classes, new_views = [], [], [], [], []
    for (cls, lbl, obj), idxs in grouped.items():
        enc_mean = enc[idxs].mean(dim=0, keepdim=True)
        dec_mean = dec[idxs].mean(dim=0, keepdim=True)
        enc_mean = F.normalize(enc_mean, dim=1)
        dec_mean = F.normalize(dec_mean, dim=1)
        new_enc.append(enc_mean)
        new_dec.append(dec_mean)
        new_labels.append(lbl)
        new_classes.append(cls)
        new_views.append(f"{obj}-mean")
    if new_enc:
        enc = torch.cat(new_enc, dim=0)
        dec = torch.cat(new_dec, dim=0)
    return enc, dec, new_labels, new_classes, new_views


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


def build_palette(classes: Sequence[str], args):
    if args.color_mode == "label":
        normal = args.normal_color or "#2ca02c"
        anomaly = args.anomaly_color or "#d62728"
        return {"normal": normal, "anomaly": anomaly}
    unique = sorted(set(classes))
    cmap = matplotlib.colormaps.get_cmap("tab20" if len(unique) <= 20 else "nipy_spectral")
    colors = [cmap(i / max(len(unique), 1)) for i in range(len(unique))]
    return {cls: colors[i] for i, cls in enumerate(unique)}


def draw_convex_hull(ax, coords, color, alpha=0.08, zorder=0):
    if len(coords) < 3:
        return
    try:
        hull = ConvexHull(coords)
    except Exception:
        return
    hull_pts = coords[hull.vertices]
    ax.fill(hull_pts[:, 0], hull_pts[:, 1], color=color, alpha=alpha, zorder=zorder, linewidth=0)


def scatter_axis(ax, coords, labels, classes, views, palette, args, title: str):
    coords = np.asarray(coords)
    normal_idx = [i for i, lbl in enumerate(labels) if lbl == 0]
    anomaly_idx = [i for i, lbl in enumerate(labels) if lbl == 1]
    view_markers = None
    if args.mark_views:
        marker_cycle = ["o", "s", "D", "P", "^", "v", "<", ">", "X", "*"]
        uniq_views = sorted(set(views))
        view_markers = {v: marker_cycle[i % len(marker_cycle)] for i, v in enumerate(uniq_views)}

    if args.color_mode == "class":
        for cls in sorted(set(classes)):
            idx_cls = [i for i, c in enumerate(classes) if c == cls and labels[i] == 0]
            if not idx_cls:
                continue
            pts = coords[idx_cls]
            if args.show_hull:
                draw_convex_hull(ax, pts, palette[cls], alpha=args.hull_alpha, zorder=0)
            if view_markers:
                for v in sorted(set(views[i] for i in idx_cls)):
                    idx_cv = [i for i in idx_cls if views[i] == v]
                    pts_v = coords[idx_cv]
                    ax.scatter(
                        pts_v[:, 0],
                        pts_v[:, 1],
                        s=args.point_size,
                        c=[palette[cls]],
                        alpha=0.35,
                        linewidths=0,
                        zorder=1,
                        marker=view_markers[v],
                    )
            else:
                ax.scatter(
                    pts[:, 0],
                    pts[:, 1],
                    s=args.point_size,
                    c=[palette[cls]],
                    alpha=0.35,
                    linewidths=0,
                    zorder=1,
                )
    else:
        # label mode: normals/abnormals share colors
        if normal_idx:
            ax.scatter(
                coords[normal_idx, 0],
                coords[normal_idx, 1],
                s=args.point_size,
                c=palette["normal"],
                alpha=0.35,
                linewidths=0,
                zorder=1,
                marker="o",
            )

    if anomaly_idx:
        if args.color_mode == "label":
            colors = [palette["anomaly"]] * len(anomaly_idx)
        else:
            colors = [args.anomaly_color or palette[classes[i]] for i in anomaly_idx]
        ax.scatter(
            coords[anomaly_idx, 0],
            coords[anomaly_idx, 1],
            s=args.point_size * 1.2,
            c=colors,
            alpha=1.0,
            linewidths=0.4,
            edgecolors="white",
            marker="^",
            zorder=5,
        )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    if args.show_legend:
        handles = [Line2D([0], [0], marker="o", color=color, linestyle="") for color in palette.values()]
        ax.legend(handles, palette.keys(), loc="upper right", fontsize=8, frameon=False)


def plot_side_by_side(enc_coords, dec_coords, labels, classes, views, args, out_path: Path):
    palette = build_palette(classes, args)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    scatter_axis(
        axes[0],
        enc_coords,
        labels,
        classes,
        views,
        palette,
        args,
        title="Original (encoder feature)",
    )
    scatter_axis(
        axes[1],
        dec_coords,
        labels,
        classes,
        views,
        palette,
        args,
        title="Reconstructed (decoder feature)",
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)


def parse_args():
    parser = argparse.ArgumentParser("t-SNE visualization for normal vs anomaly separation")
    parser.add_argument("--dataset", choices=["mvtec", "visa", "realiad"], default="mvtec")
    parser.add_argument("--data_root", type=str, default=None, help="Override dataset root.")
    parser.add_argument("--classes", type=str, default="", help="Comma separated subset of classes.")
    parser.add_argument("--model_path", type=str, required=True, help="Checkpoint to load (expects `model` key).")
    parser.add_argument("--encoder", type=str, default="dinov2reg_vit_base_14")
    parser.add_argument("--decoder", choices=["vit", "mamba", "hybrid"], default="mamba")
    parser.add_argument("--decode_depth", type=int, default=8)
    parser.add_argument("--hybrid_pat", type=str, default="v,m,v,m,v,m,v,m")
    parser.add_argument("--num_reg_tokens", type=int, default=4)
    parser.add_argument("--image_size", type=int, default=448)
    parser.add_argument("--crop_size", type=int, default=392)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max_normal_per_class", type=int, default=300)
    parser.add_argument("--max_anomaly_per_class", type=int, default=120)
    parser.add_argument("--max_total", type=int, default=4000, help="Global cap; 0 to disable.")
    parser.add_argument("--pca_dim", type=int, default=50)
    parser.add_argument("--perplexity", type=float, default=40.0)
    parser.add_argument("--early_exaggeration", type=float, default=16.0)
    parser.add_argument("--tsne_iter", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=400.0)
    parser.add_argument("--metric", choices=["euclidean", "cosine"], default="euclidean")
    parser.add_argument("--view_strategy", choices=["per_view", "multi_view_mean"], default="per_view")
    parser.add_argument("--mark_views", action="store_true", help="Use per-view markers (helpful for Real-IAD).")
    parser.add_argument("--realiad_resolution", type=int, default=1024)
    parser.add_argument("--output_dir", type=str, default="/home/user/project/Dinomaly/data/Visual")
    parser.add_argument("--save_prefix", type=str, default="")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--point_size", type=float, default=12.0)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--inline_labels", action="store_true", help="(Deprecated) Prototype text labels; kept for compatibility.")
    parser.add_argument("--show_legend", action="store_true")
    parser.add_argument("--anomaly_color", type=str, default="", help="Leave empty to color anomalies by class; set to a hex to override.")
    parser.add_argument("--normal_color", type=str, default="", help="Only used when color_mode=label; defaults to green.")
    parser.add_argument("--color_mode", choices=["class", "label"], default="class", help="Class palette vs normal/anomaly (green/red).")
    parser.add_argument("--show_hull", action="store_true", help="Draw convex hulls for normal points (class mode only).")
    parser.add_argument("--hull_alpha", type=float, default=0.08)
    parser.add_argument("--font_path", type=str, default="", help="Path to a TTF/OTF font (e.g., Times_New_Roman.ttf).")
    parser.add_argument("--font_name", type=str, default="Times New Roman", help="Font family name to request.")
    return parser.parse_args()


def main():
    args = parse_args()
    configure_fonts(args.font_path, args.font_name)
    set_seed(args.seed)
    data_root = Path(args.data_root) if args.data_root else default_root(args.dataset)
    data_transform, gt_transform = get_data_transforms(args.image_size, args.crop_size)
    model, device = build_model(args)

    loaders = build_loaders(args, data_root, data_transform, gt_transform)
    if not loaders:
        raise RuntimeError(f"No data loaders built for {args.dataset} under {data_root}.")

    enc_feats, dec_feats, labels, classes, views, obj_ids = collect_embeddings(model, loaders, device, args)
    if enc_feats.shape[0] == 0:
        raise RuntimeError("No samples collected; check dataset paths or sampling limits.")

    if args.dataset == "realiad":
        enc_feats, dec_feats, labels, classes, views = aggregate_multi_view(
            enc_feats, dec_feats, labels, classes, views, obj_ids, args.view_strategy
        )

    enc_np = enc_feats.numpy()
    dec_np = dec_feats.numpy()

    enc_coords = run_tsne(enc_np, args)
    dec_coords = run_tsne(dec_np, args)

    prefix = args.save_prefix or f"{args.dataset}"
    fname = f"tsne_{prefix}.png"
    out_path = Path(args.output_dir) / fname
    plot_side_by_side(enc_coords, dec_coords, labels, classes, views, args, out_path)
    print(f"Saved t-SNE figure to {out_path}")


if __name__ == "__main__":
    main()
