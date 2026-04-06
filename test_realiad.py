import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

"""
Real-IAD (https://realiad4ad.github.io/Real-IAD/) ships 30 industrial objects captured by
multiple camera views. Each category folder inside `realiad_1024/` contains OK (normal) and
NG (anomalous) samples, while the official JSON annotations (e.g. `realiad_jsons_sv/`) record
train/test splits and optional segmentation masks. This script mirrors the pipeline in
`test_mvtec.py` but adapts the data layer so we can train/evaluate Dinomaly on Real-IAD.
"""

import warnings
warnings.filterwarnings("ignore")

import json
import glob
import re
import zipfile
import cv2
import math
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

from PIL import Image

import numpy as np
import random
import argparse
import logging

from functools import partial
from torch.utils.data import DataLoader, ConcatDataset, Dataset
from sklearn.metrics import roc_auc_score, average_precision_score

# ====== 项目内模块 ======
from dataset import get_data_transforms, get_strong_transforms
from g2l_modules.g2l_model import G2LHyD, G2LHyDv2, DALRBlock   # ★ 引入 Mamba 解码块
from g2l_modules import encoder
from dinov1.utils import trunc_normal_
from g2l_modules.gfsr_blocks import (
    GFSRBlock, bMlp, Attention, LinearAttention, LinearAttention, DropoutSchedule, LinearGMLPTokenBlock,
    ConvBlock, FeatureJitter, CGB, SAB,
    NoisyGEGLUBottleneck, MaskTokenBottleneck, HFPNBottleneck, HVQBottle, RSCReconstructionHead
)
try:
    from g2l_modules.gfsr_blocks import DWConvGMLPTokenBlock as _DWConvGMLPTokenBlock
except ImportError:
    _DWConvGMLPTokenBlock = None
DWConvGMLPTokenBlock = _DWConvGMLPTokenBlock or SAB
from utils import (
    evaluation_batch, global_cosine, regional_cosine_hm_percent,
    global_cosine_hm_percent, WarmCosineScheduler,
    compute_image_score_from_heatmap, cal_anomaly_maps
)
from optimizers import StableAdamW
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

def save_support_list(save_dir, save_name, category, paths, meta=None):
    out_dir = os.path.join(save_dir, save_name, "support_lists")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{category}.txt")
    with open(out_path, "w") as f:
        if meta:
            for k, v in meta.items():
                f.write(f"# {k}={v}\n")
        for p in paths:
            f.write(str(p) + "\n")
    return out_path


def normalize(score: np.ndarray):
    """Min-max normalize a score map to [0, 1]."""
    score_min = float(score.min())
    score_max = float(score.max())
    denom = score_max - score_min
    if denom < 1e-8:
        return np.zeros_like(score, dtype=np.float32)
    return (score - score_min) / denom


def infer_realiad_defect_type(img_path: str, category: str):
    """
    尝试从路径中解析缺陷子类：
    .../<cat>/OK/... -> OK
    .../<cat>/<defect>/... -> defect
    找不到则返回父目录名。
    """
    norm = os.path.normpath(img_path)
    parts = norm.split(os.sep)
    if category in parts:
        idx = parts.index(category)
        if idx + 1 < len(parts):
            return parts[idx + 1]
    return os.path.basename(os.path.dirname(norm)) or "unknown"


def visualize_and_save(image_path: str, score_map: np.ndarray, save_dir: str, alpha: float = 0.5, extra_save_dirs=None):
    """Resize+normalize score map, overlay JET heatmap onto the original image, and save."""
    image = cv2.imread(image_path)
    if image is None:
        return None

    h, w = image.shape[:2]
    score_resized = cv2.resize(score_map, (w, h))
    score_norm = normalize(score_resized)
    score_uint8 = np.uint8(np.clip(score_norm * 255.0, 0, 255))
    heatmap = cv2.applyColorMap(score_uint8, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 1 - alpha, heatmap, alpha, 0)

    os.makedirs(save_dir, exist_ok=True)
    out_path = os.path.join(save_dir, os.path.basename(image_path))
    cv2.imwrite(out_path, overlay)

    if extra_save_dirs:
        for extra_dir in extra_save_dirs:
            if not extra_dir:
                continue
            os.makedirs(extra_dir, exist_ok=True)
            cv2.imwrite(os.path.join(extra_dir, os.path.basename(image_path)), overlay)
    return score_norm


def visualize_realiad_results(
    model,
    test_data_list,
    item_list,
    device,
    workers: int = 0,
    save_root: str = "results_realiad",
    max_images: int = 0,
    alpha: float = 0.5,
    print_fn=print,
):
    """
    逐类跑推理并保存热力图：
      - 主目录 results_realiad/<class>/by_defect/<defect>/
      - 镜像一份到 results_realiad/<class>/imgs/
      - 计算像素级 AUROC（若掩码全零则为 nan）
    """
    model.eval()
    px_aurocs = {}
    with torch.no_grad():
        for item, dataset in zip(item_list, test_data_list):
            loader = torch.utils.data.DataLoader(
                dataset, batch_size=1, shuffle=False, num_workers=workers
            )
            preds, gts = [], []
            base_dir = os.path.join(save_root, item)
            save_dir_flat = os.path.join(base_dir, "imgs")
            save_dir_by_defect = os.path.join(base_dir, "by_defect")

            for idx, batch in enumerate(loader):
                if len(batch) == 4:
                    img, gt, _, img_path = batch
                else:
                    img, gt, _, img_path, _ = batch
                img = img.to(device)
                anomaly_map, _ = cal_anomaly_maps(*model(img), img.shape[-1])
                amap_np = anomaly_map.squeeze(1).cpu().numpy()
                gt_np = gt.squeeze(1).cpu().numpy()

                for b in range(amap_np.shape[0]):
                    defect_type = infer_realiad_defect_type(img_path[b], item)
                    target_dir = os.path.join(save_dir_by_defect, defect_type)
                    norm_map = visualize_and_save(
                        img_path[b],
                        amap_np[b],
                        target_dir,
                        alpha=alpha,
                        extra_save_dirs=[save_dir_flat],
                    )
                    if norm_map is None:
                        continue

                    mask = gt_np[b]
                    if norm_map.shape != mask.shape:
                        mask = cv2.resize(
                            mask, (norm_map.shape[1], norm_map.shape[0]), interpolation=cv2.INTER_NEAREST
                        )

                    preds.append(norm_map.reshape(-1))
                    gts.append(mask.reshape(-1))

                if max_images and (idx + 1) >= max_images:
                    break

            auroc_px = float('nan')
            if preds and gts:
                gts_flat = np.concatenate(gts).astype(np.uint8)
                preds_flat = np.concatenate(preds).astype(np.float32)
                if np.unique(gts_flat).size > 1:
                    auroc_px = roc_auc_score(gts_flat, preds_flat)
            px_aurocs[item] = auroc_px

            if print_fn is not None:
                if math.isnan(auroc_px):
                    print_fn(f"[visualize] {item}: px-AUROC=nan (skipped)")
                else:
                    print_fn(f"[visualize] {item}: px-AUROC={auroc_px:.4f}")

    valid = [v for v in px_aurocs.values() if not math.isnan(v)]
    if print_fn is not None and valid:
        print_fn(f"[visualize] mean px-AUROC={np.mean(valid):.4f}")
    return px_aurocs


# ------------------------- Real-IAD 数据辅助 -------------------------
_VIEW_PATTERNS_CACHE = {}
_JSON_CACHE = {}
_ZIP_NAME_CACHE = {}
_UNPACKED_CATEGORIES = set()


def _discover_realiad_classes(root, user_items=None):
    """
    自动扫描 realiad_1024/ 下的类别名，既支持已解压文件夹，也支持 <cls>.zip。
    用户可通过 --items 手动指定（逗号分隔）；优先使用用户配置。
    """
    if user_items:
        items = [s.strip() for s in user_items.split(',') if s.strip()]
        if not items:
            raise ValueError("--items 提供内容为空，请检查参数。")
        return items

    img_root = os.path.join(root, 'realiad_1024')
    if not os.path.isdir(img_root):
        raise FileNotFoundError(f"未找到 realiad_1024 目录：{img_root}")

    names = set()
    for entry in os.listdir(img_root):
        path = os.path.join(img_root, entry)
        if entry.endswith('.zip'):
            names.add(os.path.splitext(entry)[0])
        elif os.path.isdir(path):
            names.add(entry)
    items = sorted(names)
    if not items:
        raise RuntimeError(f"在 {img_root} 未找到任何类别，请确认数据已解压或放置正确。")
    return items

def _try_get(d: dict, keys, default=None):
    for k in keys:
        if isinstance(d, dict) and k in d:
            return d[k]
    return default

def _label_to_int(lbl):
    try:
        v = int(lbl)
        return 0 if v <= 0 else 1
    except Exception:
        pass
    s = str(lbl).strip().upper()
    if s in ('', '0', 'OK', 'NORMAL', 'GOOD'):
        return 0
    if s in ('1', 'NG', 'ANOMALY', 'DEFECT', 'BAD'):
        return 1
    # Real-IAD 中除 OK 外的 anomaly_class 都视作缺陷
    return 1

def _normalize_split(v, default='test'):
    if v is None:
        return default
    s = str(v).strip().lower()
    if s in ('train', 'training', 'tr'):
        return 'train'
    if s in ('test', 'testing', 'te', 'val', 'valid', 'validation'):
        return 'test'
    return default

def _infer_realiad_group_id(path_or_rel):
    """
    Heuristic: strip view token (e.g., C1/C2) from filename and keep parent folders.
    This groups multi-view images belonging to the same instance.
    """
    if not path_or_rel:
        return None
    norm = os.path.normpath(str(path_or_rel))
    base = os.path.basename(norm)
    stem = os.path.splitext(base)[0]
    stem = re.sub(r'(^|[_\-])C[1-9]([_\-]|$)', r'\1', stem)
    stem = re.sub(r'[_\-]+', '_', stem).strip('_-')
    parent = os.path.dirname(norm)
    if parent:
        return os.path.join(parent, stem)
    return stem

def _is_view(path_or_name, view='C1'):
    if view in (None, 'ANY'):
        return True
    name = os.path.basename(str(path_or_name))
    pat = _VIEW_PATTERNS_CACHE.get(view)
    if pat is None:
        pat = re.compile(rf'(^|[_\-]){re.escape(view)}([_\-]|\.|$)')
        _VIEW_PATTERNS_CACHE[view] = pat
    return bool(pat.search(name))

def _resolve_image_path(root, category, rel_path):
    if rel_path is None:
        return None
    _ensure_category_unpacked(root, category)
    if os.path.isabs(rel_path) and os.path.exists(rel_path):
        return rel_path
    candidates = [
        os.path.join(root, 'realiad_1024', category, rel_path),
        os.path.join(root, 'realiad_1024', rel_path),
        os.path.join(root, rel_path),
    ]
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    raise FileNotFoundError(f"Cannot locate image '{rel_path}' under {root}")

def _resolve_mask_path(root, category, rel_path):
    if rel_path is None:
        return None
    _ensure_category_unpacked(root, category)
    if os.path.isabs(rel_path) and os.path.exists(rel_path):
        return rel_path
    candidates = [
        os.path.join(root, 'realiad_1024', category, rel_path),
        os.path.join(root, 'realiad_1024', rel_path),
        os.path.join(root, 'realiad_gts_1024', category, rel_path),
        os.path.join(root, 'realiad_gts_1024', rel_path),
        os.path.join(root, 'realiad_masks', category, rel_path),
        os.path.join(root, 'realiad_masks', rel_path),
        os.path.join(root, rel_path),
    ]
    for cand in candidates:
        if os.path.exists(cand):
            return cand
    return None

def _ensure_category_unpacked(root, category):
    key = (os.path.abspath(root), category)
    base_dir = os.path.join(root, 'realiad_1024', category)
    if os.path.isdir(base_dir):
        return base_dir
    if key in _UNPACKED_CATEGORIES:
        return base_dir if os.path.isdir(base_dir) else None
    zip_path = os.path.join(root, 'realiad_1024', f'{category}.zip')
    if os.path.isfile(zip_path):
        try:
            with zipfile.ZipFile(zip_path) as zf:
                zf.extractall(os.path.join(root, 'realiad_1024'))
            _UNPACKED_CATEGORIES.add(key)
        except zipfile.BadZipFile:
            warnings.warn(f"Zip file损坏：{zip_path}")
    return base_dir if os.path.isdir(base_dir) else None

def _read_realiad_json_obj(root, json_dir, category):
    if not json_dir:
        raise FileNotFoundError("json_dir is empty")
    base = json_dir if os.path.isabs(json_dir) else os.path.join(root, json_dir)
    fname = f'{category}.json'

    # 直接指向单个 json 文件
    if base.endswith('.json') and os.path.isfile(base):
        with open(base, 'r') as f:
            return json.load(f)

    # 目录模式
    if os.path.isdir(base):
        fp = os.path.join(base, fname)
        if os.path.isfile(fp):
            with open(fp, 'r') as f:
                return json.load(f)

    # zip 候选
    zip_candidates = []
    if base.endswith('.zip'):
        zip_candidates.append(base)
    else:
        zip_candidates.append(base + '.zip')
        base_norm = os.path.normpath(json_dir)
        if base_norm:
            top = base_norm.split(os.sep)[0]
            if top:
                zip_candidates.append(os.path.join(root, top + '.zip'))
        zip_candidates.append(os.path.join(root, 'realiad_jsons.zip'))
        zip_candidates.append(os.path.join(root, 'realiad_jsons_sv.zip'))

    seen = set()
    for zip_path in zip_candidates:
        if not zip_path or not os.path.isfile(zip_path) or zip_path in seen:
            continue
        seen.add(zip_path)
        cache = _ZIP_NAME_CACHE.setdefault(zip_path, {})
        inner = cache.get(fname)
        if inner is None:
            try:
                with zipfile.ZipFile(zip_path) as zf:
                    for name in zf.namelist():
                        base_name = os.path.basename(name.rstrip('/'))
                        if base_name == fname:
                            cache[fname] = name
                            inner = name
                            break
                    else:
                        cache[fname] = None
            except zipfile.BadZipFile:
                continue
        if inner:
            with zipfile.ZipFile(zip_path) as zf:
                with zf.open(inner) as f:
                    return json.load(f)

    raise FileNotFoundError(f"未找到 {category} 的 JSON（json_dir={json_dir}）")

def _parse_realiad_items(data):
    items = []
    if isinstance(data, dict) and any(k in data for k in ('train','test','val','validation','training','testing')):
        for split_key, entries in data.items():
            split_std = _normalize_split(split_key)
            if not isinstance(entries, list):
                continue
            for it in entries:
                img = _try_get(it, ['image','image_path','img','path'])
                if not img:
                    continue
                y = _label_to_int(_try_get(it, ['is_anomaly','anomaly','y','label','cls','anomaly_class'], 0))
                msk = _try_get(it, ['mask','mask_path','gt','anno','ann'])
                vw  = _try_get(it, ['view','camera','cam'])
                grp = _try_get(it, ['sample_id','instance_id','group_id','group','sample','sample_name','object_id',
                                    'image_id','img_id','id'])
                items.append(dict(image=img, label=y, mask=msk, split=split_std, view=vw, group=grp))
    elif isinstance(data, list):
        for it in data:
            img = _try_get(it, ['image','image_path','img','path'])
            if not img:
                continue
            y = _label_to_int(_try_get(it, ['is_anomaly','anomaly','y','label','cls','anomaly_class'], 0))
            msk = _try_get(it, ['mask','mask_path','gt','anno','ann'])
            spl = _normalize_split(_try_get(it, ['split','set','phase'], 'test'))
            vw  = _try_get(it, ['view','camera','cam'])
            grp = _try_get(it, ['sample_id','instance_id','group_id','group','sample','sample_name','object_id',
                                'image_id','img_id','id'])
            items.append(dict(image=img, label=y, mask=msk, split=spl, view=vw, group=grp))
    return items

def _load_realiad_records(root, json_dir, category):
    cache_key = (os.path.abspath(root), json_dir, category)
    if cache_key in _JSON_CACHE:
        return _JSON_CACHE[cache_key]
    data = _read_realiad_json_obj(root, json_dir, category)
    items = _parse_realiad_items(data)
    _JSON_CACHE[cache_key] = items
    return items

def _collect_by_glob(img_root, category, view='ANY', split=None):
    cat_dir = os.path.join(img_root, category)
    _ensure_category_unpacked(os.path.dirname(img_root), category)
    ok_dir = os.path.join(cat_dir, 'OK')
    ng_dir = os.path.join(cat_dir, 'NG')

    def _all_imgs(d):
        if not os.path.isdir(d):
            return []
        exts = ('*.jpg', '*.jpeg', '*.png', '*.bmp', '*.webp')
        files = []
        for ext in exts:
            files.extend(glob.glob(os.path.join(d, '**', ext), recursive=True))
        return sorted(f for f in files if not f.lower().endswith('.zip'))

    def _view_ok(path):
        if view in (None, 'ANY'):
            return True
        token = f"_{view}_"
        return token in os.path.basename(path)

    ok = [p for p in _all_imgs(ok_dir) if _view_ok(p)]
    ng = [p for p in _all_imgs(ng_dir) if _view_ok(p)]
    items = []
    if split in (None, 'train'):
        for p in ok:
            items.append(dict(image=p, label=0, split='train', view=view))
    if split in (None, 'test'):
        for p in ok:
            items.append(dict(image=p, label=0, split='test', view=view))
        for p in ng:
            items.append(dict(image=p, label=1, split='test', view=view))
    return items

class RealIADSupportDataset(Dataset):
    """Few-shot support sampler: pick K normal images per category for training/calibration."""
    def __init__(self, root, category, shots=1, view='ANY',
                 transform=None, seed=123, json_dir='realiad_jsons/realiad_jsons_sv',
                 strict_json: bool = False, strict_split: bool = False):
        super().__init__()
        self.root = root
        self.transform = transform
        self.category = category

        img_root = os.path.join(root, 'realiad_1024')
        items = None
        try:
            items = _load_realiad_records(root, json_dir, category)
        except FileNotFoundError:
            if strict_json or strict_split:
                raise
            items = None

        def _select_ok(item_list, require_train=True, filter_view=True):
            if not item_list:
                return []
            ok_list = [it for it in item_list if it['label'] == 0]
            if require_train:
                ok_list = [it for it in ok_list if _normalize_split(it.get('split', 'train')) == 'train']
            if filter_view and view not in (None, 'ANY'):
                ok_list = [it for it in ok_list if _is_view(it['image'], view)]
            return ok_list

        pool = []
        if items:
            ok_train = _select_ok(items, require_train=True, filter_view=True)
            if not ok_train and not strict_split:
                ok_train = _select_ok(items, require_train=False, filter_view=True)
            if not ok_train and view not in (None, 'ANY'):
                warnings.warn(f"[{category}] 未找到视角 {view} 的 train OK 图像，自动回退到 ALL 视角。")
                ok_train = _select_ok(items, require_train=True, filter_view=False)
                if not ok_train and not strict_split:
                    ok_train = _select_ok(items, require_train=False, filter_view=False)
            if strict_split and not ok_train:
                raise RuntimeError(f"[{category}] strict_split enabled but no train OK images found.")
            pool = [_resolve_image_path(root, category, it['image']) for it in ok_train]
        else:
            candidates = _collect_by_glob(img_root, category, view, split='train')
            ok_items = [it for it in candidates if it['label'] == 0]
            pool = [it['image'] for it in ok_items]

        if len(pool) == 0 and view not in (None, 'ANY'):
            warnings.warn(f"[{category}] 未找到视角 {view} 的 OK 图像，自动回退到 ALL 视角。")
            if items:
                ok_items = _select_ok(items, require_train=True, filter_view=False)
                if not ok_items and not strict_split:
                    ok_items = _select_ok(items, require_train=False, filter_view=False)
                if strict_split and not ok_items:
                    raise RuntimeError(f"[{category}] strict_split enabled but no train OK images found (ALL views).")
                pool = [_resolve_image_path(root, category, it['image']) for it in ok_items]
            else:
                fallback = _collect_by_glob(img_root, category, None, split='train')
                ok_items = [it for it in fallback if it['label'] == 0]
                pool = [it['image'] for it in ok_items]

        assert len(pool) > 0, f"[{category}] no OK images found even after fallback"
        rng = random.Random(seed + hash(category) % 10007)
        self.paths = rng.sample(pool, shots) if len(pool) >= shots else pool
        self.labels = [0] * len(self.paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

class RealIADTestDataset(Dataset):
    """Evaluation split (OK + NG) filtered by view token."""
    def __init__(self, root, category, view='ANY',
                 transform=None, gt_transform=None,
                 json_dir='realiad_jsons/realiad_jsons_sv',
                 exclude_paths=None,
                 strict_json: bool = False, strict_split: bool = False):
        super().__init__()
        self.root = root
        self.transform = transform
        self.gt_transform = gt_transform
        self.category = category
        self.exclude = set(exclude_paths or [])

        img_root = os.path.join(root, 'realiad_1024')
        items = None
        try:
            items = _load_realiad_records(root, json_dir, category)
        except FileNotFoundError:
            if strict_json or strict_split:
                raise
            items = None

        def _build_from_items(item_list, filter_view=True):
            if not item_list:
                return [], [], [], []
            sel = [it for it in item_list if _normalize_split(it.get('split','test')) == 'test']
            if not sel:
                if strict_split:
                    raise RuntimeError(f"[{category}] strict_split enabled but no test split found.")
                sel = item_list
            if filter_view and view not in (None, 'ANY'):
                sel = [it for it in sel if _is_view(it['image'], view)]
            paths, labels, masks, groups = [], [], [], []
            for it in sel:
                img_path = _resolve_image_path(root, category, it['image'])
                if img_path in self.exclude:
                    continue
                mask_path = _resolve_mask_path(root, category, it.get('mask'))
                paths.append(img_path)
                labels.append(int(it['label']))
                masks.append(mask_path)
                grp = it.get('group') or _infer_realiad_group_id(it.get('image'))
                groups.append(grp if grp is not None else _infer_realiad_group_id(img_path))
            return paths, labels, masks, groups

        if items:
            paths, labels, masks, groups = _build_from_items(items, filter_view=True)
        else:
            raw = _collect_by_glob(img_root, category, view, split='test')
            paths, labels, masks, groups = [], [], [], []
            for it in raw:
                if it['split'] != 'test':
                    continue
                if it['image'] in self.exclude:
                    continue
                paths.append(it['image'])
                labels.append(int(it['label']))
                masks.append(None)
                groups.append(_infer_realiad_group_id(it['image']))

        label_set = set(labels)
        if (len(paths) == 0 or len(label_set) < 2) and view not in (None, 'ANY'):
            msg = "[{}] 测试集中视角 {} {}，自动回退 ALL 视角。".format(
                category,
                view,
                "无样本" if len(paths) == 0 else "缺少正负样本"
            )
            warnings.warn(msg)
            if items:
                paths, labels, masks, groups = _build_from_items(items, filter_view=False)
            else:
                raw = _collect_by_glob(img_root, category, None, split='test')
                paths, labels, masks, groups = [], [], [], []
                for it in raw:
                    if it['split'] != 'test':
                        continue
                    if it['image'] in self.exclude:
                        continue
                    paths.append(it['image'])
                    labels.append(int(it['label']))
                    masks.append(None)
                    groups.append(_infer_realiad_group_id(it['image']))
            label_set = set(labels)

        if strict_split and len(paths) == 0:
            raise RuntimeError(f"[{category}] strict_split enabled but no test samples after filtering.")

        if len(label_set) < 2:
            warnings.warn(f"[{category}] 测试集中仅包含单一标签，相关指标可能为 NaN（view={view}).")

        self.paths = paths
        self.labels = labels
        self.masks = masks
        self.group_ids = groups

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        else:
            img = torch.from_numpy(np.array(img)).permute(2,0,1).float() / 255.0

        mask_path = self.masks[idx]
        gt = None
        if mask_path is not None and os.path.isfile(mask_path):
            mask = Image.open(mask_path).convert('L')
            if self.gt_transform:
                gt = self.gt_transform(mask)
                if isinstance(gt, torch.Tensor) and gt.ndim == 2:
                    gt = gt.unsqueeze(0)
            else:
                gt = torch.from_numpy((np.array(mask) > 0).astype(np.float32))
                if gt.ndim == 2:
                    gt = gt.unsqueeze(0)
        if gt is None:
            _, H, W = img.shape
            gt = torch.zeros(1, H, W, dtype=torch.float32)
        else:
            _, H, W = img.shape
            if gt.shape[-2:] != (H, W):
                gt = F.interpolate(gt.unsqueeze(0), size=(H, W), mode='nearest').squeeze(0)

        label = self.labels[idx]
        group_id = self.group_ids[idx] if self.group_ids else None
        return img, gt, label, path, group_id


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
    total_iters = 50000
    image_size = 448
    crop_size  = 392
    setup_seed(args.seed)
    data_transform, gt_transform = get_data_transforms(image_size, crop_size)

    train_data_list, test_data_list = [], []

    # ---------- Few-shot 设置 ----------
    shots_per_class = args.shots if args.shots in (1, 2, 4) else 1

    for i, item in enumerate(item_list):
        support = RealIADSupportDataset(
            root=args.data_path,
            category=item,
            shots=shots_per_class,
            view=args.view,
            transform=data_transform,
            seed=args.seed + i,
            json_dir=args.json_dir,
            strict_json=args.strict_json,
            strict_split=args.strict_split
        )
        if getattr(args, "save_support_list", True):
            meta = {
                "category": item,
                "shots": shots_per_class,
                "seed": args.seed + i,
                "view": args.view,
                "json_dir": args.json_dir,
                "strict_json": args.strict_json,
                "strict_split": args.strict_split,
            }
            list_path = save_support_list(args.save_dir, args.save_name, item, support.paths, meta=meta)
            print_fn(f"[support] saved list: {list_path}")
        test_data = RealIADTestDataset(
            root=args.data_path,
            category=item,
            view=args.view,
            transform=data_transform,
            gt_transform=gt_transform,
            json_dir=args.json_dir,
            exclude_paths=getattr(support, "paths", None),
            strict_json=args.strict_json,
            strict_split=args.strict_split
        )

        train_data_list.append(support)
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
            return DWConvGMLPTokenBlock(
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

    # ========== 可选：加载已有权重 / 仅可视化 ==========
    if getattr(args, "load_ckpt", None):
        ckpt_path = os.path.expanduser(args.load_ckpt)
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        state_dict = state.get('model', state) if isinstance(state, dict) else state
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print_fn(f"[ckpt] loaded from {ckpt_path} | missing={len(missing)} | unexpected={len(unexpected)}")

    if getattr(args, "visualize_only", False):
        visualize_realiad_results(
            model=model,
            test_data_list=test_data_list,
            item_list=item_list,
            device=device,
            workers=args.workers,
            save_root=args.vis_root,
            max_images=args.vis_limit,
            alpha=0.5,
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

           # --- 定期评测 & 日志：先自增，再统一用 it（自然数） ---
            it += 1

            # 评测触发：从 eval_start 开始，每隔 eval_every 次评一次
            if (it >= args.eval_start) and ((it - args.eval_start) % args.eval_every == 0):
                print_fn(f'======== EVAL @ step {it} ========')

                # Step-1：选择图像级聚合器
                agg_fn = None
                if args.hotspot_gem:
                    from functools import partial
                    agg_fn = partial(compute_image_score_from_heatmap,
                                    k_ratio=args.topk_ratio, p=args.gem_p, fg_q=args.fg_q)
                elif args.i_agg == 'max_topk':
                    from utils import image_score_max_topk
                    from functools import partial
                    agg_fn = partial(image_score_max_topk,
                                    alpha=args.i_alpha, top_percent=args.i_top_percent)
                # else: 使用默认 max（evaluation_batch 内部处理）

                sample_metrics = args.eval_mode in ('sample', 'both')
                auroc_sp_list, ap_sp_list, f1_sp_list = [], [], []
                auroc_px_list, ap_px_list, f1_px_list, aupro_px_list = [], [], [], []
                auroc_sa_list, ap_sa_list, f1_sa_list = [], [], []

                # === 逐类评测 ===
                for (item, test_data, calib_data) in zip(item_list, test_data_list, train_data_list):
                    test_dataloader = torch.utils.data.DataLoader(
                        test_data, batch_size=batch_size, shuffle=False, num_workers=args.workers
                    )

                    results = evaluation_batch(
                        model, test_dataloader, device,
                        max_ratio=0.01, resize_mask=256,
                        aggregator=agg_fn,
                        flip_tta=args.flip_tta,
                        z_norm=args.z_norm,
                        z_calib_dataset=calib_data,  # few-shot 正常图用于 μ/σ
                        postproc=args.postproc,
                        topk_ratio=args.topk_ratio, gem_p=args.gem_p, fg_q=args.fg_q,
                        post_k=args.post_k, post_iters=args.post_iters,
                        rot_tta=args.rot_tta, ms_tta=args.ms_tta,
                        px_metrics=args.px_metrics,
                        sample_metrics=sample_metrics
                    )

                    if sample_metrics:
                        auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px, auroc_sa, ap_sa, f1_sa = results
                        auroc_sa_list.append(auroc_sa); ap_sa_list.append(ap_sa); f1_sa_list.append(f1_sa)
                    else:
                        auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results
                    auroc_sp_list.append(auroc_sp); ap_sp_list.append(ap_sp); f1_sp_list.append(f1_sp)
                    auroc_px_list.append(auroc_px); ap_px_list.append(ap_px); f1_px_list.append(f1_px); aupro_px_list.append(aupro_px)

                    if args.eval_mode == 'image':
                        print_fn('{}: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, '
                                'P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                                    item, auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px))
                    elif args.eval_mode == 'sample':
                        print_fn('{}: S-Auroc:{:.4f}, S-AP:{:.4f}, S-F1:{:.4f}, '
                                'P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                                    item, auroc_sa, ap_sa, f1_sa, auroc_px, ap_px, f1_px, aupro_px))
                    else:
                        print_fn('{}: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, '
                                'S-Auroc:{:.4f}, S-AP:{:.4f}, S-F1:{:.4f}, '
                                'P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                                    item, auroc_sp, ap_sp, f1_sp, auroc_sa, ap_sa, f1_sa,
                                    auroc_px, ap_px, f1_px, aupro_px))

                if args.eval_mode == 'image':
                    print_fn('Mean: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, '
                            'P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                                np.mean(auroc_sp_list), np.mean(ap_sp_list), np.mean(f1_sp_list),
                                np.mean(auroc_px_list), np.mean(ap_px_list), np.mean(f1_px_list), np.mean(aupro_px_list)))
                elif args.eval_mode == 'sample':
                    print_fn('Mean: S-Auroc:{:.4f}, S-AP:{:.4f}, S-F1:{:.4f}, '
                            'P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                                np.mean(auroc_sa_list), np.mean(ap_sa_list), np.mean(f1_sa_list),
                                np.mean(auroc_px_list), np.mean(ap_px_list), np.mean(f1_px_list), np.mean(aupro_px_list)))
                else:
                    print_fn('Mean: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, '
                            'S-Auroc:{:.4f}, S-AP:{:.4f}, S-F1:{:.4f}, '
                            'P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                                np.mean(auroc_sp_list), np.mean(ap_sp_list), np.mean(f1_sp_list),
                                np.mean(auroc_sa_list), np.mean(ap_sa_list), np.mean(f1_sa_list),
                                np.mean(auroc_px_list), np.mean(ap_px_list), np.mean(f1_px_list), np.mean(aupro_px_list)))
                model.train()

            # 训练日志：同样用 it，自然对齐
            if (it % args.log_every) == 0:
                print_fn('iter [{}/{}], loss:{:.4f}'.format(it, total_iters, np.mean(loss_list)))
                loss_list = []

            # 终止条件
            if it == total_iters:
                break

    # ========== Save final checkpoint ==========
    ckpt_path = args.export_ckpt
    if ckpt_path:
        ckpt_path = os.path.expanduser(ckpt_path)
        ckpt_dir = os.path.dirname(ckpt_path)
        if ckpt_dir:
            os.makedirs(ckpt_dir, exist_ok=True)
    else:
        ckpt_dir = os.path.join(args.save_dir, args.save_name, "ckpts_final")
        os.makedirs(ckpt_dir, exist_ok=True)
        ckpt_path = os.path.join(ckpt_dir, "final.pth")
    torch.save(model.state_dict(), ckpt_path)
    print_fn(f"[ckpt] saved to {ckpt_path}")

    # ========== Optional visualization after training ==========
    if getattr(args, "save_visuals", False):
        visualize_realiad_results(
            model=model,
            test_data_list=test_data_list,
            item_list=item_list,
            device=device,
            workers=args.workers,
            save_root=args.vis_root,
            max_images=args.vis_limit,
            alpha=0.5,
            print_fn=print_fn
        )

    return


# ------------------------- 入口 -------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Few-shot Dinomaly (Real-IAD view-selectable) with ViT/Mamba/Hybrid decoders'
    )
    parser.add_argument('--data_path', type=str, default='/data/IAD/Real-IAD/',
                        help='Real-IAD 根目录，需包含 realiad_1024/ 与 json/zip')
    parser.add_argument('--json_dir', type=str, default='realiad_jsons.zip',
                        help='JSON 注释来源：可为目录或 zip，相对 data_path')
    parser.add_argument('--view', type=str, default='ANY',
                        help='相机视角（C1...C5/ANY）；缺少正负样本会自动回退')
    parser.add_argument('--items', type=str, default='',
                        help='逗号分隔的类别名，留空则自动扫描 realiad_1024')
    parser.add_argument('--save_dir',  type=str, default='/home/user/project/Dinomaly/data/RealIAD')
    parser.add_argument('--save_name', type=str, default='realiad_fs2_default')
    parser.add_argument('--load_ckpt', type=str, default='',
                        help='若指定则在训练/可视化前加载该 ckpt')
    parser.add_argument('--export_ckpt', type=str, default='',
                        help='若非空则将最终模型保存到该路径，否则默认保存到 save_dir/save_name/ckpts_final/final.pth')
    parser.add_argument('--visualize_only', action='store_true',
                        help='只跑可视化与像素 AUROC，跳过训练')
    parser.add_argument('--save_visuals', action='store_true',
                        help='训练完成后跑可视化并保存')
    parser.add_argument('--vis_root', type=str, default='results_realiad',
                        help='可视化输出根目录（默认 results_realiad/<class>/...）')
    parser.add_argument('--vis_limit', type=int, default=0,
                        help='每类最多保存多少张可视化，0 表示不限制')
    parser.add_argument('--strict_json', action='store_true',
                        help='Require JSON annotations; disable glob fallback')
    parser.add_argument('--strict_split', action='store_true',
                        help='Require official train/test split; do not fall back to full set')
    parser.add_argument('--eval_mode', type=str, default='image', choices=['image', 'sample', 'both'],
                        help='Evaluation mode: image-level, sample-level, or both')
    parser.add_argument('--no_save_support_list', action='store_false', dest='save_support_list',
                        help='Disable saving selected support image lists')
    parser.set_defaults(save_support_list=True)

    parser.add_argument('--shots', type=int, default=4, choices=[1, 2, 4],
                        help='Few-shot support size per class (only OK images)')
    parser.add_argument('--seed', type=int, default=5, help='Random seed for few-shot sampling & training')
    parser.add_argument('--cuda', type=int, default=0, help='CUDA device index')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=4)

    parser.add_argument('--decoder', type=str, default='hybrid',
                        choices=['vit', 'mamba', 'hybrid'],
                        help='Decoder type: pure ViT / pure Mamba / hybrid (Mamba↔ViT)')
    parser.add_argument('--decoder_attn', type=str, default='linear', choices=['linear', 'full'],
                        help='attention type for ViT blocks in decoder')
    parser.add_argument('--hybrid_pat', type=str, default='v,v,v,v,m,m,m,m',
                        help='Pattern for hybrid across N blocks, m= Mamba, v= ViT')
    parser.add_argument('--decode_depth', type=int, default=8,
                        help='Number of decoder blocks (default 8 matches reference configs)')
    parser.add_argument('--encoder', type=str, default='dinov2reg_vit_base_14',
                        help='{dinov2reg_vit_small_14 | dinov2reg_vit_base_14 | dinov2reg_vit_large_14 | ...}')
    parser.add_argument('--bn_dw_kernel', type=int, default=5)
    parser.add_argument('--bn_drop', type=float, default=0.1)

    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--lr_end', type=float, default=2e-4)
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--eval_start', type=int, default=40000,
                        help='First evaluation step; evaluate every eval_every steps afterwards')
    parser.add_argument('--eval_every', type=int, default=5000)
    parser.add_argument('--log_every',  type=int, default=500)

    parser.add_argument('--hotspot_gem', action='store_true',
                        help='Use Hotspot-GeM aggregator for image-level scores')
    parser.add_argument('--topk_ratio', type=float, default=0.032,
                        help='Top-k ratio for hotspot selection, e.g., 0.02 = 2% pixels')
    parser.add_argument('--gem_p', type=float, default=8.0,
                        help='GeM pooling p for image-level aggregation')
    parser.add_argument('--fg_q', type=int, default=64,
                        help='Foreground gate quantile on token norm (0-100)')
    parser.add_argument('--flip_tta', action='store_true',
                        help='Use horizontal flip TTA at eval time')
    parser.add_argument('--rot_tta', type=str, default='none',
                        help="Rotation TTA policy: 'none' | 'all' | comma-separated class list")
    parser.add_argument('--ms_tta', type=str, default='1.0,1.10',
                        help="Multi-scale TTA factors, e.g. '0.75,1.0' (1.0 is implicit)")
    parser.add_argument('--z_norm', action='store_true',
                        help='Class-wise z-normalization for image scores at eval')
    parser.add_argument('--postproc', action='store_true',
                        help='Morphological smoothing on heatmaps (Step-2 post-processing)')
    parser.add_argument('--post_k', type=int, default=3, help='Morph kernel size (odd)')
    parser.add_argument('--post_iters', type=int, default=1, help='Morph iterations')
    parser.add_argument('--i_agg', type=str, default='max', choices=['max', 'max_topk'],
                        help='Image-level aggregator: max (default) or max_topk')
    parser.add_argument('--i_alpha', type=float, default=0.6,
                        help='Alpha for max_topk aggregator (0..1)')
    parser.add_argument('--i_top_percent', type=float, default=5.0,
                        help='Top-percent for max_topk aggregator (e.g. 5 → top 5%% pixels)')
    parser.add_argument('--no_px_metrics', action='store_false', dest='px_metrics',
                        help='禁用像素级指标（默认开启）')
    parser.set_defaults(px_metrics=True)

    parser.add_argument('--bottleneck_variant', type=str, default='gmlp_dw',
        choices=['none', 'cgb', 'pg_cgb', 'cgb_lora', 'ms', 'pixel', 'cgbv2', 'osb', 'sgb','gmlp_dw','gmlp_lin',
                 'noisy_geglu', 'jitter', 'mask', 'hvq', 'rsc'],
        help='Bottleneck block variant for G2LHyD')
    parser.add_argument('--bn_drop_start',  type=float, default=0.0)
    parser.add_argument('--bn_drop_end',    type=float, default=0.2)
    parser.add_argument('--bn_drop_warmup', type=int,   default=1000)
    parser.add_argument('--jitter_sigma', type=float, default=0.10)
    parser.add_argument('--mask_ratio',   type=float, default=0.30)
    parser.add_argument('--vq_k',         type=int,   default=512)
    parser.add_argument('--vq_beta',      type=float, default=0.25)
    parser.add_argument('--vq_ema_decay', type=float, default=0.99)
    parser.add_argument('--vq_lambda',    type=float, default=0.25, help='HVQ loss weight')

    parser.add_argument('--mamba_ks', type=str, default='5,7',
                        help="Mamba local kernel sizes, e.g. '5,7'")
    parser.add_argument('--mamba_dilations', type=str, default='1,2',
                        help="Dilations for the two local kernels, e.g. '1,2'")
    parser.add_argument('--mamba_dw_kernel', type=int, default=0,
                        help='Override kernel sizes with a single depth-wise kernel (>0)')
    parser.add_argument('--mamba_add_local3', type=int, default=1, choices=[0,1],
                        help='Whether to add an extra 3x3 branch inside Mamba blocks')
    parser.add_argument('--mamba_add_asym', type=int, default=1, choices=[0,1],
                        help='Whether to add asymmetric 1xk/kx1 branches')
    parser.add_argument('--mamba_num_hss', type=int, default=3,
                        help='Number of HSS scanning branches inside Mamba blocks')

    # +++ Mamba 扫描方式与方向 +++
    parser.add_argument('--mamba_scan_method', type=str, default='hilbert',
                        choices=['hilbert', 'sweep', 'scan'],
                        help="Scan method for HSS module in Mamba (hilbert/sweep/scan).")
    parser.add_argument('--mamba_scan_dirs', type=int, default=8,
                        choices=[2, 4, 8],
                        help="Number of scan directions for HSS module (2/4/8).")

    args = parser.parse_args()

    if not os.path.isdir(args.data_path):
        raise FileNotFoundError(
            f"data_path '{args.data_path}' 不存在，请使用 --data_path 指向包含 realiad_1024 的目录"
        )

    item_list = _discover_realiad_classes(args.data_path, args.items)

    logger   = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info
    print_fn(f"Loaded {len(item_list)} classes: {item_list}")

    device = f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu'
    print_fn(device)

    train(item_list)
