import os
from typing import Iterable, Sequence, Tuple

import cv2
import numpy as np


def normalize(pred):
    """Min-max normalize prediction map to [0, 1] with zero-division guard."""
    pred = np.asarray(pred, dtype=np.float32)
    pred_min = float(np.min(pred))
    pred_max = float(np.max(pred))
    denom = pred_max - pred_min
    if denom <= 1e-12:
        return np.zeros_like(pred, dtype=np.float32)
    return (pred - pred_min) / denom


def apply_ad_scoremap(image_rgb, scoremap, alpha=0.5):
    """
    Build JET heatmap from scoremap and alpha-blend with RGB image.
    Returns: heatmap_rgb, overlay_rgb
    """
    np_image = np.asarray(image_rgb, dtype=np.float32)
    scoremap = np.asarray(scoremap, dtype=np.float32)
    alpha = float(np.clip(alpha, 0.0, 1.0))

    h, w = np_image.shape[:2]
    if scoremap.shape[:2] != (h, w):
        scoremap = cv2.resize(scoremap, (w, h), interpolation=cv2.INTER_LINEAR)

    score_uint8 = np.uint8(np.clip(normalize(scoremap) * 255.0, 0, 255))
    heatmap_rgb = cv2.cvtColor(cv2.applyColorMap(score_uint8, cv2.COLORMAP_JET), cv2.COLOR_BGR2RGB)
    overlay_rgb = (alpha * np_image + (1.0 - alpha) * heatmap_rgb.astype(np.float32)).astype(np.uint8)
    return heatmap_rgb, overlay_rgb


def _infer_defect_type(img_path: str) -> str:
    parts = os.path.normpath(str(img_path)).split(os.sep)
    for anchor in ("test", "train"):
        if anchor in parts:
            anchor_idx = parts.index(anchor)
            if anchor_idx + 1 < len(parts):
                return parts[anchor_idx + 1]
    return os.path.basename(os.path.dirname(str(img_path))) or "unknown"


def _parse_img_size(img_size) -> Tuple[int, int]:
    if isinstance(img_size, int):
        return int(img_size), int(img_size)
    if isinstance(img_size, (tuple, list)) and len(img_size) == 2:
        return int(img_size[0]), int(img_size[1])
    raise ValueError(f"Unsupported img_size: {img_size}")


def _as_path_list(img_paths) -> Sequence[str]:
    if isinstance(img_paths, (str, os.PathLike)):
        return [str(img_paths)]
    if isinstance(img_paths, np.ndarray):
        return [str(p) for p in img_paths.tolist()]
    if isinstance(img_paths, Iterable):
        return [str(p) for p in img_paths]
    raise ValueError("img_paths must be a path string or iterable of paths.")


def _as_scoremaps(anomaly_map) -> np.ndarray:
    maps = np.asarray(anomaly_map, dtype=np.float32)
    if maps.ndim == 2:
        return maps[None, ...]
    if maps.ndim == 3:
        return maps
    if maps.ndim == 4 and maps.shape[1] == 1:
        return maps[:, 0, ...]
    raise ValueError(f"Unsupported anomaly_map shape: {maps.shape}")


def visualizer(img_paths, anomaly_map, img_size, save_path, cls_name, alpha=0.5):
    """
    Save original/heatmap/overlay with structure:
    <save_path>/imgs/<cls_name>/<defect_type>/{original,heatmap,overlay}/filename
    """
    path_list = _as_path_list(img_paths)
    scoremaps = _as_scoremaps(anomaly_map)
    dst_h, dst_w = _parse_img_size(img_size)
    n = min(len(path_list), scoremaps.shape[0])

    for i in range(n):
        image_path = path_list[i]
        image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            continue
        image_bgr = cv2.resize(image_bgr, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR)
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

        scoremap = scoremaps[i]
        if scoremap.shape[:2] != (dst_h, dst_w):
            scoremap = cv2.resize(scoremap, (dst_w, dst_h), interpolation=cv2.INTER_LINEAR)
        heatmap_rgb, overlay_rgb = apply_ad_scoremap(image_rgb, scoremap, alpha=alpha)

        defect_type = _infer_defect_type(image_path)
        file_name = os.path.basename(image_path)
        root = os.path.join(save_path, "imgs", cls_name, defect_type)
        original_dir = os.path.join(root, "original")
        heatmap_dir = os.path.join(root, "heatmap")
        overlay_dir = os.path.join(root, "overlay")
        os.makedirs(original_dir, exist_ok=True)
        os.makedirs(heatmap_dir, exist_ok=True)
        os.makedirs(overlay_dir, exist_ok=True)

        cv2.imwrite(os.path.join(original_dir, file_name), image_bgr)
        cv2.imwrite(
            os.path.join(heatmap_dir, file_name),
            cv2.cvtColor(heatmap_rgb, cv2.COLOR_RGB2BGR),
        )
        cv2.imwrite(
            os.path.join(overlay_dir, file_name),
            cv2.cvtColor(overlay_rgb, cv2.COLOR_RGB2BGR),
        )
