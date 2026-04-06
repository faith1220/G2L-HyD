# Dinomaly/models_mambaAD/hilbert_scan.py
import math
import torch
from functools import lru_cache

def _rotate(p, n, rx, ry):
    if ry == 0:
        if rx == 1:
            p[0] = n - 1 - p[0]
            p[1] = n - 1 - p[1]
        p[0], p[1] = p[1], p[0]
    return p

def _d2xy(n, d):
    """Hilbert distance -> (x,y) for n=2^order."""
    p = [0, 0]
    t = d
    s = 1
    while s < n:
        rx = 1 & (t // 2)
        ry = 1 & (t ^ rx)
        p = _rotate(p, s, rx, ry)
        p[0] += s * rx
        p[1] += s * ry
        t //= 4
        s *= 2
    return p[0], p[1]

@lru_cache(maxsize=128)
def _hilbert_order_indices(h, w, device='cpu'):
    """返回 [L] 的线性 index（Hilbert 顺序）。允许非 2^n，自动在 2^k 方格上生成后裁剪。"""
    n = 1 << math.ceil(math.log2(max(h, w)))
    coords = []
    for d in range(n * n):
        x, y = _d2xy(n, d)
        if x < w and y < h:
            coords.append((y, x))   # row-major (y,x)
        if len(coords) == h * w:
            break
    idx = torch.tensor([y * w + x for (y, x) in coords], dtype=torch.long, device=device)
    return idx  # [L]

def _row_sweep_indices(h, w, device='cpu'):
    return torch.arange(h*w, device=device, dtype=torch.long)

def _col_scan_indices(h, w, device='cpu'):
    grid = torch.arange(h*w, device=device).view(h, w)
    return grid.t().contiguous().view(-1)

def _apply_direction(index_2d, h, w, direction):
    """根据方向对 (y*w+x) 的平面索引序列做翻转/旋转。"""
    grid = index_2d.view(-1)  # [L]
    if direction == 'fwd':
        return grid
    # 还原为 (y,x)，做旋转翻转，再映射回线性索引
    y = grid // w
    x = grid % w
    if direction == 'rev':
        y = (h - 1) - y; x = (w - 1) - x
    elif direction == 'wh_fwd':
        x, y = y, x
        h, w = w, h
    elif direction == 'wh_rev':
        x, y = y, x
        h, w = w, h
        y = (h - 1) - y; x = (w - 1) - x
    elif direction == 'rot90_fwd':
        x, y = (h - 1) - y, x
        h, w = w, h
    elif direction == 'rot90_rev':
        x, y = y, (w - 1) - x
        h, w = w, h
    elif direction == 'wh_rot90_fwd':
        # Transpose + Rot90 = H-flip
        x, y = (w - 1) - x, y
    elif direction == 'wh_rot90_rev':
        # Transpose + Rot90 + Rev = V-flip
        x, y = x, (h - 1) - y
    else:
        return grid
    return (y * w + x).view(-1)

def _get_indices(method, h, w, device):
    if method == 'hilbert':
        return _hilbert_order_indices(h, w, device=device)
    elif method == 'sweep':
        return _row_sweep_indices(h, w, device=device)
    elif method == 'scan':
        return _col_scan_indices(h, w, device=device)
    else:
        return _hilbert_order_indices(h, w, device=device)

SCAN_DIRECTIONS = {
    2: ['fwd', 'rev'],
    4: ['fwd', 'rev', 'rot90_fwd', 'rot90_rev'],
    8: ['fwd', 'rev', 'wh_fwd', 'wh_rev', 'rot90_fwd', 'rot90_rev', 'wh_rot90_fwd', 'wh_rot90_rev']
}

def get_scan_directions(num_directions=8):
    """
    Returns a list of scan directions.
    """
    if num_directions not in SCAN_DIRECTIONS:
        raise ValueError(f"Unsupported number of directions: {num_directions}. Supported values are {list(SCAN_DIRECTIONS.keys())}.")
    return SCAN_DIRECTIONS[num_directions]

def scan_2d_to_1d(x, method='hilbert', direction='fwd'):
    """
    x: [B, C, H, W] -> seq: [B, L, C], L=H*W
    """
    B, C, H, W = x.shape
    base_idx = _get_indices(method, H, W, x.device)
    idx = _apply_direction(base_idx, H, W, direction)
    flat = x.flatten(2)  # [B, C, L]
    seq  = flat[:, :, idx]  # gather
    return seq.transpose(1, 2).contiguous()  # [B, L, C]

def descan_1d_to_2d(seq, H, W, method='hilbert', direction='fwd'):
    """
    seq: [B, L, C] -> [B, C, H, W]
    """
    B, L, C = seq.shape
    base_idx = _get_indices(method, H, W, seq.device)
    idx = _apply_direction(base_idx, H, W, direction)
    flat = torch.zeros(B, C, L, device=seq.device, dtype=seq.dtype)
    flat[:, :, idx] = seq.transpose(1, 2)
    return flat.view(B, C, H, W)
