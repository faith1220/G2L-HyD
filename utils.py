from dataset import get_data_transforms
from torchvision.datasets import ImageFolder
import numpy as np
from torch.utils.data import DataLoader
from dataset import MVTecDataset
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score, f1_score, recall_score, accuracy_score, precision_recall_curve, \
    average_precision_score
import cv2
import matplotlib.pyplot as plt
from sklearn.metrics import auc
from skimage import measure
import pandas as pd
from numpy import ndarray
from statistics import mean
from scipy.ndimage import gaussian_filter, binary_dilation
import os
from functools import partial
import pickle
# utils/agg.py
import torch.nn as nn
import math, torch
import torch.nn.functional as F
from g2l_modules.memknn_head import build_memory_bank, knn_heatmap
from typing import Callable, Optional, Tuple


###################agg Hotspot-GeM + 前景门控###############################
@torch.no_grad()
def _gem(x, p=6.0, eps=1e-6):
    # x: [B,H,W] or [B,N] in [0,1]
    x = x.clamp(min=eps).pow(p)
    x = x.mean(dim=(-1, -2)) if x.dim() == 3 else x.mean(dim=-1)
    return x.pow(1.0 / p)

@torch.no_grad()
def _topk_mask(hm, k_ratio=0.02):
    # hm: [B,1,H,W] in [0,1]
    B, _, H, W = hm.shape
    K = max(1, int(H * W * k_ratio))
    flat = hm.view(B, -1)
    thresh = torch.topk(flat, K, dim=1, largest=True).values[:, -1].unsqueeze(-1)
    return (flat >= thresh).view(B, 1, H, W).float()

@torch.no_grad()
def _fg_gate_from_tokens(enc_patch_tokens, H, W, q=70):
    """
    enc_patch_tokens: [B, H*W, C]（可用 encoder 或 decoder 的 patch tokens）
    没有 tokens 时，调用方传 None，会退化为热图分位数门控。
    """
    if enc_patch_tokens is None:
        return None
    B, N, C = enc_patch_tokens.shape
    assert H * W == N, f"Token grid mismatch: H*W={H*W}, N={N}"
    m = enc_patch_tokens.norm(p=2, dim=-1)             # [B,N]
    thr = torch.quantile(m, q/100.0, dim=1, keepdim=True)
    return (m >= thr).float().view(B, 1, H, W)         # [B,1,H,W]

@torch.no_grad()
def compute_image_score_from_heatmap(hm, enc_patch_tokens=None, H=None, W=None,
                                     k_ratio=0.02, p=6.0, fg_q=70):
    """
    hm: [B, 1, Hh, Wh] in [0,1]（若上游是多通道，这里会先归并为单通道）
    enc_patch_tokens: [B, H*W, C] 或 None（用于前景门控；None 时退化为 hm 分位数门控）
    H, W: patch 网格高宽；若 enc_patch_tokens 给到则可从 N=H*W 推出；否则可为 None
    return: image_score [B]，用于 I-AUROC / AP
    """
    # --- 统一到单通道 ---
    if hm.dtype != torch.float32:
        hm = hm.float()
    B, C, Hh, Wh = hm.shape
    if C != 1:
        # 用通道最大值收敛为单通道（比 mean 更利于“热点”）
        hm = hm.amax(dim=1, keepdim=True)
        C = 1  # 仅记账，无实际用途

    # --- 前景门控：优先 token 范数；否则用热图分位数 ---
    if enc_patch_tokens is not None:
        if (H is None) or (W is None):
            N = enc_patch_tokens.shape[1]
            side = int(N ** 0.5)
            assert side * side == N, "patch tokens not square"
            H = W = side
        fg = _fg_gate_from_tokens(enc_patch_tokens, H, W, q=fg_q)  # [B,1,H,W]
        if (H, W) != (Hh, Wh):
            fg = F.interpolate(fg, size=(Hh, Wh), mode='bilinear', align_corners=False)
    else:
        # 用热图自身的分位数（例如 70 分位）近似前景
        flat = hm.view(B, -1)                                    # [B, Hh*Wh]
        thr = torch.quantile(flat, fg_q / 100.0, dim=1, keepdim=True)  # [B,1]
        fg  = (flat >= thr).float().view(B, 1, Hh, Wh)                 # [B,1,Hh,Wh]

    # --- Hotspot（top-k%） ---
    hot  = _topk_mask(hm, k_ratio=k_ratio)                       # [B,1,Hh,Wh]
    gate = fg * hot                                              # 前景 ∩ 热点

    # --- 归一化（防空集） ---
    denom  = gate.sum(dim=(-1, -2, -3), keepdim=True).clamp_min(1.0)
    hm_sel = (hm * gate) / denom

    # --- GeM 聚合 → 图像级分数 ---
    score = _gem(hm_sel.squeeze(1), p=p)                         # [B]
    return score
@torch.no_grad()
def image_score_max_topk(hm, alpha=0.6, top_percent=5.0, **kwargs):
    """
    hm: [B,1,H,W] in [0,1]
    return: [B] = alpha*max + (1-alpha)*mean(top k%)
    """
    if hm.dtype != torch.float32:
        hm = hm.float()
    if hm.shape[1] > 1:
        hm = hm.amax(dim=1, keepdim=True)   # 规范为单通道

    B = hm.shape[0]
    flat = hm.view(B, -1)                   # [B,HW]
    maxv = flat.max(dim=1).values
    k = max(1, int(flat.shape[1] * (top_percent / 100.0)))
    topk_mean = torch.topk(flat, k, dim=1, largest=True).values.mean(dim=1)
    return alpha * maxv + (1.0 - alpha) * topk_mean


@torch.no_grad()
def _morph_smooth(hm, k=3, iters=1):
    """
    简单形态学开运算 + 轻度均值平滑；去毛刺、抑制孤立高分点
    hm: [B,1,H,W]，值域 [0,1]
    """
    x = hm
    for _ in range(iters):
        erode = -F.max_pool2d(-x, kernel_size=k, stride=1, padding=k//2)
        x = F.max_pool2d(erode, kernel_size=k, stride=1, padding=k//2)
    x = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    return x

###########################################################################

def modify_grad(x, inds, factor=0.):
    inds = inds.expand_as(x)
    x[inds] *= factor
    return x


def modify_grad_v2(x, factor):
    factor = factor.expand_as(x)
    x *= factor
    return x


def global_cosine(a, b, stop_grad=True):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        if stop_grad:
            loss += torch.mean(1 - cos_loss(a[item].view(a[item].shape[0], -1).detach(),
                                            b[item].view(b[item].shape[0], -1)))
        else:
            loss += torch.mean(1 - cos_loss(a[item].view(a[item].shape[0], -1),
                                            b[item].view(b[item].shape[0], -1)))
    loss = loss / len(a)
    return loss


def global_cosine_hm(a, b, alpha=1., factor=0.):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        a_ = a[item].detach()
        b_ = b[item]
        with torch.no_grad():
            point_dist = 1 - cos_loss(a_, b_).unsqueeze(1)
        mean_dist = point_dist.mean()
        std_dist = point_dist.reshape(-1).std()

        loss += torch.mean(1 - cos_loss(a_.reshape(a_.shape[0], -1),
                                        b_.reshape(b_.shape[0], -1)))
        thresh = mean_dist + alpha * std_dist
        partial_func = partial(modify_grad, inds=point_dist < thresh, factor=factor)
        b_.register_hook(partial_func)
    # loss = loss / len(a)
    return loss


def global_cosine_hm_percent(a, b, p=0.9, factor=0., weights=None):
    cos_loss = torch.nn.CosineSimilarity()
    loss_list = []
    for item in range(len(a)):
        a_ = a[item].detach()
        b_ = b[item]
        with torch.no_grad():
            point_dist = 1 - cos_loss(a_, b_).unsqueeze(1)
        # mean_dist = point_dist.mean()
        # std_dist = point_dist.reshape(-1).std()
        thresh = torch.topk(point_dist.reshape(-1), k=int(point_dist.numel() * (1 - p)))[0][-1]

        loss_list.append(torch.mean(1 - cos_loss(a_.reshape(a_.shape[0], -1),
                                                 b_.reshape(b_.shape[0], -1))))

        partial_func = partial(modify_grad, inds=point_dist < thresh, factor=factor)
        b_.register_hook(partial_func)

    if weights is None:
        loss = sum(loss_list) / len(a)
    else:
        w = weights
        if not torch.is_tensor(w):
            w = torch.tensor(w, dtype=loss_list[0].dtype, device=loss_list[0].device)
        w = w.flatten()
        if w.numel() != len(loss_list):
            raise ValueError(f"weights length {w.numel()} != num_groups {len(loss_list)}")
        w = w / (w.sum() + 1e-6)
        loss = sum(w[i] * loss_list[i] for i in range(len(loss_list)))
    return loss


def regional_cosine_hm_percent(a, b, p=0.9, factor=0.):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        a_ = a[item].detach()
        b_ = b[item]
        point_dist = 1 - cos_loss(a_, b_).unsqueeze(1)
        # mean_dist = point_dist.mean()
        # std_dist = point_dist.reshape(-1).std()
        thresh = torch.topk(point_dist.reshape(-1), k=int(point_dist.numel() * (1 - p)))[0][-1]

        loss += point_dist.mean()

        partial_func = partial(modify_grad, inds=point_dist < thresh, factor=factor)
        b_.register_hook(partial_func)

    loss = loss / len(a)
    return loss


def global_cosine_focal(a, b, p=0.9, alpha=2., min_grad=0.):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        a_ = a[item].detach()
        b_ = b[item]
        with torch.no_grad():
            point_dist = 1 - cos_loss(a_, b_).unsqueeze(1).detach()

        if p < 1.:
            thresh = torch.topk(point_dist.reshape(-1), k=int(point_dist.numel() * (1 - p)))[0][-1]
        else:
            thresh = point_dist.max()
        focal_factor = torch.clip(point_dist, max=thresh) / thresh

        focal_factor = focal_factor ** alpha
        focal_factor = torch.clip(focal_factor, min=min_grad)

        loss += torch.mean(1 - cos_loss(a_.reshape(a_.shape[0], -1),
                                        b_.reshape(b_.shape[0], -1)))

        partial_func = partial(modify_grad_v2, factor=focal_factor)
        b_.register_hook(partial_func)

    return loss


def regional_cosine_focal(a, b, p=0.9, alpha=2.):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        a_ = a[item].detach()
        b_ = b[item]

        point_dist = 1 - cos_loss(a_, b_).unsqueeze(1)
        if p < 1.:
            thresh = torch.topk(point_dist.reshape(-1), k=int(point_dist.numel() * (1 - p)))[0][-1]
        else:
            thresh = point_dist.max()
        focal_factor = torch.clip(point_dist, max=thresh) / thresh
        focal_factor = focal_factor ** alpha

        loss += (point_dist * focal_factor.detach()).mean()

    return loss


def regional_cosine_hm(a, b, p=0.9):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        a_ = a[item].detach()
        b_ = b[item]

        point_dist = 1 - cos_loss(a_, b_).unsqueeze(1)
        thresh = torch.topk(point_dist.reshape(-1), k=int(point_dist.numel() * (1 - p)))[0][-1]

        L = point_dist[point_dist >= thresh]
        loss += L.mean()

    return loss


def region_cosine(a, b, stop_grad=True):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        loss += 1 - cos_loss(a[item].detach(), b[item]).mean()
    return loss


def cal_anomaly_map(fs_list, ft_list, out_size=224, amap_mode='add', norm_factor=None):
    if not isinstance(out_size, tuple):
        out_size = (out_size, out_size)
    if amap_mode == 'mul':
        anomaly_map = np.ones(out_size)
    else:
        anomaly_map = np.zeros(out_size)

    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        if norm_factor is not None:
            a_map = 0.1 * (a_map - norm_factor[0][i]) / (norm_factor[1][i] - norm_factor[0][i])

        a_map = a_map[0, 0, :, :].to('cpu').detach().numpy()
        a_map_list.append(a_map)
        if amap_mode == 'mul':
            anomaly_map *= a_map
        else:
            anomaly_map += a_map
    return anomaly_map, a_map_list


def _as_weight_tensor(w, ref):
    if torch.is_tensor(w):
        t = w
    else:
        t = torch.tensor(w, dtype=ref.dtype, device=ref.device)
    if t.ndim == 0:
        return t.view(1, 1, 1, 1)
    if t.ndim == 1:
        if t.numel() == ref.shape[0]:
            return t.view(-1, 1, 1, 1)
        return t.view(1, 1, 1, 1)
    if t.ndim == 2:
        return t.view(t.shape[0], 1, 1, 1)
    if t.ndim == 3:
        return t.unsqueeze(1)
    return t


def _confidence_from_map(a_map, mode: str, topk_ratio: float, eps: float):
    flat = a_map.flatten(2)  # [B,1,HW]
    mode = (mode or "topk_mean").lower()
    if mode == "topk_mean":
        k = max(1, int(flat.shape[-1] * topk_ratio))
        topk = torch.topk(flat, k=k, dim=-1).values
        return topk.mean(dim=-1).squeeze(1)
    if mode in ("var", "variance"):
        return flat.var(dim=-1).squeeze(1)
    if mode == "entropy":
        p = flat / (flat.sum(dim=-1, keepdim=True) + eps)
        return -(p * (p + eps).log()).sum(dim=-1).squeeze(1)
    return flat.mean(dim=-1).squeeze(1)


def cal_anomaly_maps(
    fs_list,
    ft_list,
    out_size=224,
    weights=None,
    eps: float = 1e-6,
    fusion_mode: str = 'mean',
    fusion_confidence: str = 'topk_mean',
    fusion_topk: float = 0.02,
    fusion_temp: float = 0.07,
    fusion_gate=None,
):
    """
    计算异常图并融合。
    
    Args:
        fs_list: 编码器特征列表
        ft_list: 解码器特征列表
        out_size: 输出尺寸
        weights: 权重列表 (用于 weighted_mean / learnable)
        eps: 数值稳定性常数
        fusion_mode: 融合模式 ('mean', 'max', 'weighted_mean', 'conf_softmax', 'learnable', 'spatial_gate')
        fusion_confidence: conf_softmax 的置信度方式
        fusion_topk: conf_softmax 的 top-k 比例
        fusion_temp: softmax 温度
        fusion_gate: spatial_gate 的可学习门控模块
    
    Returns:
        anomaly_map: [B, 1, H, W] 融合后的异常图
        a_map_list: 各组的异常图列表
    """
    if not isinstance(out_size, tuple):
        out_size = (out_size, out_size)

    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size, mode='bilinear', align_corners=True)
        a_map_list.append(a_map)
    
    weight_list = None
    if weights is not None:
        if torch.is_tensor(weights):
            if weights.ndim == 1:
                weight_list = [weights[i] for i in range(weights.numel())]
            elif weights.ndim == 2:
                weight_list = [weights[:, i] for i in range(weights.shape[1])]
            else:
                weight_list = [weights[i] for i in range(weights.shape[0])]
        else:
            weight_list = list(weights)
        if len(weight_list) != len(a_map_list):
            raise ValueError(f"weights length {len(weight_list)} != num_groups {len(a_map_list)}")

    # 根据融合模式选择不同的融合策略
    if fusion_mode == 'mean':
        anomaly_map = torch.cat(a_map_list, dim=1).sum(dim=1, keepdim=True) / len(a_map_list)

    elif fusion_mode in ('weighted_mean', 'learnable'):
        weighted_maps = []
        weight_sum = None
        for i, a_map in enumerate(a_map_list):
            if weight_list is not None and i < len(weight_list) and weight_list[i] is not None:
                w = _as_weight_tensor(weight_list[i], a_map).to(a_map.device, a_map.dtype)
                weighted_maps.append(a_map * w)
                weight_sum = w if weight_sum is None else weight_sum + w
            else:
                weighted_maps.append(a_map)
        anomaly_map = torch.cat(weighted_maps, dim=1).sum(dim=1, keepdim=True)
        if weight_sum is not None:
            anomaly_map = anomaly_map / (weight_sum + eps)
        else:
            anomaly_map = anomaly_map / len(a_map_list)

    elif fusion_mode == 'conf_softmax':
        conf_list = [_confidence_from_map(a_map, fusion_confidence, fusion_topk, eps) for a_map in a_map_list]
        conf = torch.stack(conf_list, dim=1)  # [B,G]
        temp = fusion_temp if fusion_temp > 1e-6 else 1e-6
        weights_dyn = torch.softmax(conf / temp, dim=1)
        weighted_maps = []
        weight_sum = None
        for i, a_map in enumerate(a_map_list):
            w = _as_weight_tensor(weights_dyn[:, i], a_map).to(a_map.device, a_map.dtype)
            weighted_maps.append(a_map * w)
            weight_sum = w if weight_sum is None else weight_sum + w
        anomaly_map = torch.cat(weighted_maps, dim=1).sum(dim=1, keepdim=True)
        anomaly_map = anomaly_map / (weight_sum + eps)

    elif fusion_mode == 'max':
        anomaly_map = torch.stack(a_map_list, dim=1).max(dim=1, keepdim=True)[0]

    elif fusion_mode == 'spatial_gate':
        if fusion_gate is None:
            raise ValueError("fusion_gate is required when fusion_mode='spatial_gate'")
        stacked = torch.cat(a_map_list, dim=1)  # [B,G,H,W]
        logits = fusion_gate(stacked)
        temp = fusion_temp if fusion_temp > 1e-6 else 1e-6
        weights_gate = torch.softmax(logits / temp, dim=1)
        anomaly_map = (stacked * weights_gate).sum(dim=1, keepdim=True)

    else:
        raise ValueError(f"Unknown fusion_mode: {fusion_mode}")
    
    return anomaly_map, a_map_list



def map_normalization(fs_list, ft_list, start=0.5, end=0.95):
    start_list = []
    end_list = []
    with torch.no_grad():
        for i in range(len(ft_list)):
            fs = fs_list[i]
            ft = ft_list[i]
            a_map = 1 - F.cosine_similarity(fs, ft)
            start_list.append(torch.quantile(a_map, q=start).item())
            end_list.append(torch.quantile(a_map, q=end).item())

    return [start_list, end_list]


def cal_anomaly_map_v2(fs_list, ft_list, out_size=224, amap_mode='add'):
    a_map_list = []
    for i in range(len(ft_list)):
        fs = fs_list[i]
        ft = ft_list[i]
        a_map = 1 - F.cosine_similarity(fs, ft)
        a_map = torch.unsqueeze(a_map, dim=1)
        a_map = F.interpolate(a_map, size=out_size // 4, mode='bilinear', align_corners=False)
        a_map_list.append(a_map)

    anomaly_map = torch.stack(a_map_list, dim=-1).sum(-1)
    anomaly_map = F.interpolate(anomaly_map, size=out_size, mode='bilinear', align_corners=False)
    anomaly_map = anomaly_map[0, 0, :, :].to('cpu').detach().numpy()

    return anomaly_map, a_map_list


def show_cam_on_image(img, anomaly_map):
    cam = np.float32(anomaly_map) / 255 + np.float32(img) / 255
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


def min_max_norm(image):
    a_min, a_max = image.min(), image.max()
    return (image - a_min) / (a_max - a_min)


def create_custom_colormap():
    # Create a custom blue-to-red colormap
    # This is a linear interpolation between blue, white, and red
    c1 = np.array([255, 0, 0])   # Blue
    c2 = np.array([255, 255, 255]) # White
    c3 = np.array([0, 0, 255])   # Red
    
    colormap = np.zeros((256, 1, 3), dtype=np.uint8)
    for i in range(256):
        if i < 128:
            # Blue to White
            val = i / 128.0
            color = (1 - val) * c1 + val * c2
        else:
            # White to Red
            val = (i - 128) / 128.0
            color = (1 - val) * c2 + val * c3
        colormap[i, 0, :] = color.astype(np.uint8)
    return colormap


def create_viridis_colormap():
    """
    创建 viridis colormap（蓝绿黄色调），与 Figure_A2 风格一致。
    这是一个感知均匀的 colormap，适合异常检测热力图可视化。
    """
    # Viridis colormap 的关键颜色点 (BGR 格式，因为 OpenCV 使用 BGR)
    # 从深蓝 -> 青色 -> 绿色 -> 黄色
    colors = [
        (68, 1, 84),      # 深紫蓝 (索引 0)
        (72, 40, 120),    # 紫色
        (62, 74, 137),    # 蓝紫
        (49, 104, 142),   # 蓝色
        (38, 130, 142),   # 青蓝
        (31, 158, 137),   # 青色
        (53, 183, 121),   # 青绿
        (109, 205, 89),   # 绿色
        (180, 222, 44),   # 黄绿
        (253, 231, 37),   # 黄色 (索引 255)
    ]
    
    colormap = np.zeros((256, 1, 3), dtype=np.uint8)
    n_colors = len(colors)
    
    for i in range(256):
        # 计算在颜色列表中的位置
        pos = i / 255.0 * (n_colors - 1)
        idx = int(pos)
        frac = pos - idx
        
        if idx >= n_colors - 1:
            color = colors[-1]
        else:
            # 线性插值
            c1 = np.array(colors[idx])
            c2 = np.array(colors[idx + 1])
            color = (1 - frac) * c1 + frac * c2
        
        # BGR 格式 (OpenCV)
        colormap[i, 0, :] = color.astype(np.uint8)
    
    return colormap


def cvt2heatmap(gray, colormap=None):
    """
    将灰度图转换为热力图。
    
    Args:
        gray: 灰度图像 (0-255)
        colormap: 颜色映射，可以是:
            - None: 使用 viridis colormap (默认，与 Figure_A2 风格一致)
            - cv2.COLORMAP_* 常量
            - 自定义 colormap 数组
    """
    if colormap is None:
        colormap = create_viridis_colormap()
    heatmap = cv2.applyColorMap(np.uint8(gray), colormap)
    return heatmap


def return_best_thr(y_true, y_score):
    precs, recs, thrs = precision_recall_curve(y_true, y_score)

    f1s = 2 * precs * recs / (precs + recs + 1e-7)
    f1s = f1s[:-1]
    thrs = thrs[~np.isnan(f1s)]
    f1s = f1s[~np.isnan(f1s)]
    best_thr = thrs[np.argmax(f1s)]
    return best_thr


def f1_score_max(y_true, y_score):
    precs, recs, thrs = precision_recall_curve(y_true, y_score)

    f1s = 2 * precs * recs / (precs + recs + 1e-7)
    f1s = f1s[:-1]
    return f1s.max()


def specificity_score(y_true, y_score):
    y_true = np.array(y_true)
    y_score = np.array(y_score)

    TN = (y_true[y_score == 0] == 0).sum()
    N = (y_true == 0).sum()
    return TN / N


def evaluation(model, dataloader, device, _class_=None, calc_pro=True, norm_factor=None, feature_used='all',
               max_ratio=0):
    model.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    aupro_list = []

    with torch.no_grad():
        for img, gt, label, _ in dataloader:
            img = img.to(device)

            en, de = model(img)

            if feature_used == 'trained':
                anomaly_map, _ = cal_anomaly_map(en[3:], de[3:], img.shape[-1], amap_mode='a', norm_factor=norm_factor)
            elif feature_used == 'freezed':
                anomaly_map, _ = cal_anomaly_map(en[:3], de[:3], img.shape[-1], amap_mode='a', norm_factor=norm_factor)
            else:
                anomaly_map, _ = cal_anomaly_map(en, de, img.shape[-1], amap_mode='a', norm_factor=norm_factor)
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            # gt[gt > 0.5] = 1
            # gt[gt <= 0.5] = 0
            gt = gt.bool()

            if calc_pro:
                if label.item() != 0:
                    aupro_list.append(compute_pro(gt.squeeze(0).cpu().numpy().astype(int),
                                                  anomaly_map[np.newaxis, :, :]))
            gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
            pr_list_px.extend(anomaly_map.ravel())
            gt_list_sp.append(np.max(gt.cpu().numpy().astype(int)))
            if max_ratio <= 0:
                sp_score = anomaly_map.max()
            else:
                anomaly_map = anomaly_map.ravel()
                sp_score = np.sort(anomaly_map)[-int(anomaly_map.shape[0] * max_ratio):]
                sp_score = sp_score.mean()
            pr_list_sp.append(sp_score)
        auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 4)
        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 4)

    return auroc_px, auroc_sp, round(np.mean(aupro_list), 4)


# def evaluation_batch(model, dataloader, device, _class_=None, max_ratio=0, resize_mask=None):
    # model.eval()
    # gt_list_px = []
    # pr_list_px = []
    # gt_list_sp = []
    # pr_list_sp = []
    # gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

    # with torch.no_grad():
    #     for img, gt, label, img_path in dataloader:
    #         img = img.to(device)
    #         # starter.record()
    #         output = model(img)
    #         # ender.record()
    #         # torch.cuda.synchronize()
    #         # curr_time = starter.elapsed_time(ender)
    #         en, de = output[0], output[1]

    #         anomaly_map, _ = cal_anomaly_maps(en, de, img.shape[-1])
    #         # anomaly_map = anomaly_map - anomaly_map.mean(dim=[1, 2, 3]).view(-1, 1, 1, 1)

    #         if resize_mask is not None:
    #             anomaly_map = F.interpolate(anomaly_map, size=resize_mask, mode='bilinear', align_corners=False)
    #             gt = F.interpolate(gt, size=resize_mask, mode='nearest')

    #         anomaly_map = gaussian_kernel(anomaly_map)

    #         gt = gt.bool()
    #         if gt.shape[1] > 1:
    #             gt = torch.max(gt, dim=1, keepdim=True)[0]

    #         gt_list_px.append(gt)
    #         pr_list_px.append(anomaly_map)
    #         gt_list_sp.append(label)

    #         if max_ratio == 0:
    #             sp_score = torch.max(anomaly_map.flatten(1), dim=1)[0]
    #         else:
    #             anomaly_map = anomaly_map.flatten(1)
    #             sp_score = torch.sort(anomaly_map, dim=1, descending=True)[0][:, :int(anomaly_map.shape[1] * max_ratio)]
    #             sp_score = sp_score.mean(dim=1)
    #         pr_list_sp.append(sp_score)

    #     gt_list_px = torch.cat(gt_list_px, dim=0)[:, 0].cpu().numpy()
    #     pr_list_px = torch.cat(pr_list_px, dim=0)[:, 0].cpu().numpy()
    #     gt_list_sp = torch.cat(gt_list_sp).flatten().cpu().numpy()
    #     pr_list_sp = torch.cat(pr_list_sp).flatten().cpu().numpy()

    #     aupro_px = compute_pro(gt_list_px, pr_list_px)

    #     gt_list_px, pr_list_px = gt_list_px.ravel(), pr_list_px.ravel()

    #     auroc_px = roc_auc_score(gt_list_px, pr_list_px)
    #     auroc_sp = roc_auc_score(gt_list_sp, pr_list_sp)
    #     ap_px = average_precision_score(gt_list_px, pr_list_px)
    #     ap_sp = average_precision_score(gt_list_sp, pr_list_sp)

    #     f1_sp = f1_score_max(gt_list_sp, pr_list_sp)
    #     f1_px = f1_score_max(gt_list_px, pr_list_px)

    # return [auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px]


# 10月2号晚上使用这个跑出了94.95的水平
# def evaluation_batch(
#     model,
#     dataloader,
#     device,
#     _class_=None,
#     max_ratio=0,
#     resize_mask=None,
#     # ---------- 新增参数（保持兼容） ----------
#     tta_hflip: bool = False,               # [ADDED] 方案2-1：是否启用水平翻转 TTA
#     per_class_thr: float | None = None,    # [ADDED] A3：该“类/品类”的像素阈值（来自训练few-shot统计）
#     thr_mode: str = "sub"                  # [ADDED] 阈值应用方式：'sub' 减阈值并ReLU；'bin' 直接二值化
# ):
    
#     model.eval()
#     gt_list_px = []
#     pr_list_px = []
#     gt_list_sp = []
#     pr_list_sp = []
#     gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

#     # [ADDED] 将标量阈值变成张量，便于广播
#     thr_tensor = None
#     if per_class_thr is not None:
#         thr_tensor = torch.tensor(float(per_class_thr), device=device).view(1, 1, 1, 1)

#     with torch.no_grad():
#         for img, gt, label, img_path in dataloader:
#             img = img.to(device)

#             # ---------- 原图前向 ----------
#             en, de = model(img)
#             anomaly_map, _ = cal_anomaly_maps(en, de, img.shape[-1])  # [B,1,H,W]

#             # ---------- [ADDED] hflip TTA ----------
#             if tta_hflip:
#                 img_flip = torch.flip(img, dims=[-1])                  # 水平翻转输入
#                 en_f, de_f = model(img_flip)
#                 amap_flip, _ = cal_anomaly_maps(en_f, de_f, img.shape[-1])  # 翻转图的热力图
#                 amap_flip = torch.flip(amap_flip, dims=[-1])           # 翻回原方向
#                 anomaly_map = 0.5 * (anomaly_map + amap_flip)          # 与原图热力图做平均

#             # ---------- 统一 resize（若需要） ----------
#             if resize_mask is not None:
#                 anomaly_map = F.interpolate(anomaly_map, size=resize_mask, mode='bilinear', align_corners=False)
#                 gt = F.interpolate(gt, size=resize_mask, mode='nearest')

#             # ---------- [ADDED] A3：按类阈值做像素级校准 ----------
#             if thr_tensor is not None:
#                 if thr_mode == 'sub':
#                     # 减去阈值并 ReLU，保留“超过正常上界”的异常分数
#                     anomaly_map = torch.clamp(anomaly_map - thr_tensor, min=0.0)
#                 elif thr_mode == 'bin':
#                     # 直接二值化（通常不建议用于 ROC/AP，但可用于固定阈值的F1/可视化）
#                     anomaly_map = (anomaly_map > thr_tensor).float()
#                 else:
#                     raise ValueError(f"Unknown thr_mode: {thr_mode}")

#             # ---------- 平滑 ----------
#             anomaly_map = gaussian_kernel(anomaly_map)

#             # ---------- GT 处理 ----------
#             gt = gt.bool()
#             if gt.shape[1] > 1:
#                 gt = torch.max(gt, dim=1, keepdim=True)[0]

#             # ---------- 收集像素/图像级分数 ----------
#             gt_list_px.append(gt)
#             pr_list_px.append(anomaly_map)
#             gt_list_sp.append(label)

#             if max_ratio == 0:
#                 sp_score = torch.max(anomaly_map.flatten(1), dim=1)[0]
#             else:
#                 amap_flat = anomaly_map.flatten(1)
#                 k = int(amap_flat.shape[1] * max_ratio)
#                 sp_score = torch.sort(amap_flat, dim=1, descending=True)[0][:, :k].mean(dim=1)
#             pr_list_sp.append(sp_score)

#         # ---------- 堆叠并转 numpy ----------
#         gt_list_px = torch.cat(gt_list_px, dim=0)[:, 0].cpu().numpy()
#         pr_list_px = torch.cat(pr_list_px, dim=0)[:, 0].cpu().numpy()
#         gt_list_sp = torch.cat(gt_list_sp).flatten().cpu().numpy()
#         pr_list_sp = torch.cat(pr_list_sp).flatten().cpu().numpy()

#         # PRO（阈值曲线下面积）仍使用连续分数
#         aupro_px = compute_pro(gt_list_px, pr_list_px)

#         gt_list_px, pr_list_px = gt_list_px.ravel(), pr_list_px.ravel()

#         # 阈值无关指标保持不变
#         auroc_px = roc_auc_score(gt_list_px, pr_list_px)
#         auroc_sp = roc_auc_score(gt_list_sp, pr_list_sp)
#         ap_px = average_precision_score(gt_list_px, pr_list_px)
#         ap_sp = average_precision_score(gt_list_sp, pr_list_sp)

#         # F1 依旧取最优阈值（阈值无关），保持与原版可比
#         f1_sp = f1_score_max(gt_list_sp, pr_list_sp)
#         f1_px = f1_score_max(gt_list_px, pr_list_px)

#     return [auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px]

# 这是10月三号上午的版本。下一个版本是旋转TTA，多尺度TTA
# def evaluation_batch(
#     model,
#     dataloader,
#     device,
#     _class_=None,
#     max_ratio=0,
#     resize_mask=None,

#     # 兼容你原有的参数
#     tta_hflip: bool = False,
#     per_class_thr: float | None = None,
#     thr_mode: str = "sub",

#     # Step-1 / Step-2
#     aggregator=None,
#     flip_tta: bool | None = None,   # None -> 等于 tta_hflip
#     z_norm: bool = False,
#     z_calib_dataset=None,
#     postproc: bool = False,

#     # Hotspot-GeM 超参（当 aggregator=compute_image_score_from_heatmap 时使用）
#     topk_ratio: float = 0.02,
#     gem_p: float = 6.0,
#     fg_q: int = 70,
#     post_k=3, post_iters=1
# ):
#     if flip_tta is None:
#         flip_tta = tta_hflip

#     model.eval()
#     gt_list_px, pr_list_px = [], []
#     gt_list_sp, pr_list_sp = [], []

#     gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

#     # 像素级标量阈值 → 张量
#     thr_tensor = None
#     if per_class_thr is not None:
#         thr_tensor = torch.tensor(float(per_class_thr), device=device).view(1, 1, 1, 1)

#     # ---------------- 先做 z-norm 的 μ/σ 统计（若启用） ----------------
#     mu = sigma = None
#     if z_norm and (z_calib_dataset is not None):
#         calib_loader = torch.utils.data.DataLoader(
#             z_calib_dataset, batch_size=min(8, len(z_calib_dataset)),
#             shuffle=False, num_workers=0, drop_last=False
#         )
#         scores_c = []
#         with torch.no_grad():
#             for calib_img, _ in calib_loader:
#                 calib_img = calib_img.to(device)
#                 en_c, de_c = model(calib_img)
#                 amap_c, _ = cal_anomaly_maps(en_c, de_c, calib_img.shape[-1])  # [B,?,H,W]

#                 # Flip-TTA（逐像素 max）
#                 if flip_tta:
#                     calib_flip = torch.flip(calib_img, dims=[-1])
#                     en_c2, de_c2 = model(calib_flip)
#                     amap_c2, _ = cal_anomaly_maps(en_c2, de_c2, calib_img.shape[-1])
#                     amap_c2 = torch.flip(amap_c2, dims=[-1])
#                     amap_c = torch.maximum(amap_c, amap_c2)

#                 # resize
#                 if resize_mask is not None:
#                     amap_c = F.interpolate(amap_c, size=resize_mask, mode='bilinear', align_corners=False)

#                 # 像素阈值校准（可选）
#                 if thr_tensor is not None:
#                     if thr_mode == 'sub':
#                         amap_c = torch.clamp(amap_c - thr_tensor, min=0.0)
#                     elif thr_mode == 'bin':
#                         amap_c = (amap_c > thr_tensor).float()

#                 # 平滑 +（可选）形态学
#                 amap_c = gaussian_kernel(amap_c)
#                 if postproc:
#                     amap_c = _morph_smooth(amap_c, k=post_k, iters=post_iters)

#                 # ★ 规范为单通道（若多通道）
#                 if amap_c.shape[1] > 1:
#                     amap_c = amap_c.amax(dim=1, keepdim=True)

#                 # 图像级分数
#                 if aggregator is not None:
#                     img_scores_c = aggregator(
#                         hm=amap_c, enc_patch_tokens=None, H=None, W=None,
#                         k_ratio=topk_ratio, p=gem_p, fg_q=fg_q
#                     )
#                 else:
#                     amap_flat = amap_c.flatten(1)
#                     if max_ratio == 0:
#                         img_scores_c = torch.max(amap_flat, dim=1)[0]
#                     else:
#                         k = max(1, int(amap_flat.shape[1] * max_ratio))
#                         img_scores_c = torch.sort(amap_flat, dim=1, descending=True)[0][:, :k].mean(dim=1)

#                 scores_c.append(img_scores_c.detach().cpu())

#         scores_c = torch.cat(scores_c, dim=0).float()
#         mu = scores_c.mean().item()
#         sigma = scores_c.std(unbiased=False).item() + 1e-6

#     # ---------------- 正式评测 ----------------
#     with torch.no_grad():
#         for img, gt, label, img_path in dataloader:
#             img = img.to(device)

#             # 原图前向
#             en, de = model(img)
#             anomaly_map, _ = cal_anomaly_maps(en, de, img.shape[-1])  # [B,?,H,W]

#             # Flip-TTA（逐像素 max）
#             if flip_tta:
#                 img_flip = torch.flip(img, dims=[-1])
#                 en_f, de_f = model(img_flip)
#                 amap_flip, _ = cal_anomaly_maps(en_f, de_f, img.shape[-1])
#                 amap_flip = torch.flip(amap_flip, dims=[-1])
#                 anomaly_map = torch.maximum(anomaly_map, amap_flip)

#             # 统一 resize
#             if resize_mask is not None:
#                 anomaly_map = F.interpolate(anomaly_map, size=resize_mask, mode='bilinear', align_corners=False)
#                 gt = F.interpolate(gt, size=resize_mask, mode='nearest')

#             # 像素级阈值校准（可选）
#             if thr_tensor is not None:
#                 if thr_mode == 'sub':
#                     anomaly_map = torch.clamp(anomaly_map - thr_tensor, min=0.0)
#                 elif thr_mode == 'bin':
#                     anomaly_map = (anomaly_map > thr_tensor).float()

#             # 平滑 +（可选）形态学
#             anomaly_map = gaussian_kernel(anomaly_map)
#             if postproc:
#                 anomaly_map = _morph_smooth(anomaly_map, k=post_k, iters=post_iters)

#             # ★ 规范为单通道（若多通道）
#             if anomaly_map.shape[1] > 1:
#                 anomaly_map = anomaly_map.amax(dim=1, keepdim=True)

#             # GT 处理
#             gt = gt.bool()
#             if gt.shape[1] > 1:
#                 gt = torch.max(gt, dim=1, keepdim=True)[0]

#             # 像素级收集
#             gt_list_px.append(gt)
#             pr_list_px.append(anomaly_map)

#             # 图像级分数
#             if aggregator is not None:
#                 sp_score = aggregator(
#                     hm=anomaly_map, enc_patch_tokens=None, H=None, W=None,
#                     k_ratio=topk_ratio, p=gem_p, fg_q=fg_q
#                 )  # [B]
#             else:
#                 amap_flat = anomaly_map.flatten(1)
#                 if max_ratio == 0:
#                     sp_score = torch.max(amap_flat, dim=1)[0]
#                 else:
#                     k = max(1, int(amap_flat.shape[1] * max_ratio))
#                     sp_score = torch.sort(amap_flat, dim=1, descending=True)[0][:, :k].mean(dim=1)

#             # z-norm（仅图像级）
#             if (mu is not None) and (sigma is not None):
#                 sp_score = (sp_score - mu) / sigma

#             gt_list_sp.append(label)
#             pr_list_sp.append(sp_score)

#         # 统计指标
#         gt_list_px = torch.cat(gt_list_px, dim=0)[:, 0].cpu().numpy()
#         pr_list_px = torch.cat(pr_list_px, dim=0)[:, 0].cpu().numpy()
#         gt_list_sp = torch.cat(gt_list_sp).flatten().cpu().numpy()
#         pr_list_sp = torch.cat(pr_list_sp).flatten().cpu().numpy()

#         aupro_px = compute_pro(gt_list_px, pr_list_px)

#         gt_list_px, pr_list_px = gt_list_px.ravel(), pr_list_px.ravel()
#         auroc_px = roc_auc_score(gt_list_px, pr_list_px)
#         auroc_sp = roc_auc_score(gt_list_sp, pr_list_sp)
#         ap_px = average_precision_score(gt_list_px, pr_list_px)
#         ap_sp = average_precision_score(gt_list_sp, pr_list_sp)

#         f1_sp = f1_score_max(gt_list_sp, pr_list_sp)
#         f1_px = f1_score_max(gt_list_px, pr_list_px)

#     return [auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px]

# ------- 新增：Texture 白名单 -------
_MVTEC_TEXTURES = {"carpet", "grid", "leather", "tile", "wood"}

def _should_rotate(cls_name: Optional[str], policy: Optional[str]) -> bool:
    if not policy:
        return False
    s = policy.strip().lower()
    if s in ("none", "off", "0"):
        return False
    if s == "all":
        return True
    if s == "textures":
        return (cls_name or "").lower() in _MVTEC_TEXTURES
    # 支持逗号分隔的类名列表
    allow = {x.strip().lower() for x in s.split(",") if x.strip()}
    return (cls_name or "").lower() in allow

def _parse_scales(ms_tta: Optional[str]):
    if not ms_tta:
        return []
    vals = []
    for x in ms_tta.split(","):
        x = x.strip()
        if not x:
            continue
        try:
            v = float(x)
            if v <= 0:
                continue
            # 1.0 不必重复；我们会始终包含原图
            if abs(v - 1.0) < 1e-6:
                continue
            vals.append(v)
        except Exception:
            pass
    # 去重并排序（小→大）
    vals = sorted(set(vals))
    return vals

@torch.no_grad()
def _morph_smooth(hm, k=3, iters=1):
    x = hm
    for _ in range(iters):
        erode = -F.max_pool2d(-x, kernel_size=k, stride=1, padding=k//2)
        x = F.max_pool2d(erode, kernel_size=k, stride=1, padding=k//2)
    x = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
    return x

# # 10.28原本的基线。
# def evaluation_batch(
#     model,
#     dataloader,
#     device,
#     _class_=None,
#     max_ratio=0,
#     resize_mask=None,

#     # 你已有的参数
#     tta_hflip: bool = False,
#     per_class_thr: float | None = None,
#     thr_mode: str = "sub",
#     aggregator=None,
#     flip_tta: bool | None = None,
#     z_norm: bool = False,
#     z_calib_dataset=None,
#     postproc: bool = False,
#     topk_ratio: float = 0.02,
#     gem_p: float = 6.0,
#     fg_q: int = 70,
#     post_k: int = 3,
#     post_iters: int = 1,
#     # —— KNN 记忆库融合（可选）——
#     use_bank: bool = False,
#     bank_encoder=None,           # callable(images)-> {'tokens':...}或{'patch':...}
#     bank_feats: torch.Tensor | None = None,   # [M,C]
#     bank_grid: tuple | None = None,           # (h,w)，来自MemoryBank
#     bank_alpha: float = 0.30,                 # 融合权重
#     bank_topk: int = 1,                       # KNN的k
    

#     # ------- 新增：旋转与多尺度 TTA -------
#     rot_tta: str | None = None,   # 'none' | 'all' | 'textures' | 'cls1,cls2'
#     ms_tta: str | None = None     # 如 "0.75,1.0"（1.0 会被忽略，因为原图已包含）
    
# ):
#     if flip_tta is None:
#         flip_tta = tta_hflip

#     model.eval()
#     gt_list_px, pr_list_px = [], []
#     gt_list_sp, pr_list_sp = [], []
#     gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

#     # 像素阈值 → 张量
#     thr_tensor = None
#     if per_class_thr is not None:
#         thr_tensor = torch.tensor(float(per_class_thr), device=device).view(1, 1, 1, 1)

#     # ---------- 小工具：单次前向得到热图 ----------
#     @torch.no_grad()
#     def _fwd_heatmap(x):
#         en, de = model(x)
#         amap, _ = cal_anomaly_maps(en, de, x.shape[-1])  # [B,?,H,W]
#         return amap

#     # ---------- 小工具：对一批图像做 TTA（独立合并，不做笛卡尔乘积） ----------
#     def _tta_merge(img, do_flip: bool, do_rot: bool, scales: list[float]):
#         H = img.shape[-1]
#         maps = []

#         # 原图
#         amap = _fwd_heatmap(img)
#         maps.append(amap)

#         # 水平翻转（逐像素max融合）
#         if do_flip:
#             x = torch.flip(img, dims=[-1])
#             a = _fwd_heatmap(x)
#             a = torch.flip(a, dims=[-1])
#             maps.append(a)

#         # 旋转 90/180/270（只对白名单类启用）
#         if do_rot:
#             for k in (1, 2, 3):
#                 x = torch.rot90(img, k=k, dims=[-2, -1])
#                 a = _fwd_heatmap(x)
#                 a = torch.rot90(a, k=4 - k, dims=[-2, -1])
#                 maps.append(a)

#         # 多尺度（比如 0.75×），回到原分辨率再融合
#         for s in scales:
#             new_hw = int(round(H * s))
#             new_hw = max(32, new_hw)  # 保底，避免太小
#             x = F.interpolate(img, size=new_hw, mode='bilinear', align_corners=False)
#             a = _fwd_heatmap(x)
#             a = F.interpolate(a, size=H, mode='bilinear', align_corners=False)
#             maps.append(a)

#         # 逐像素 max 融合（倾向于保留热点）
#         out = maps[0]
#         for m in maps[1:]:
#             out = torch.maximum(out, m)
#         return out  # [B,?,H,W]

#     # ---------------- z-norm μ/σ 统计（如启用） ----------------
#     mu = sigma = None
#     do_rot_calib = _should_rotate(_class_, rot_tta)
#     scales = _parse_scales(ms_tta)

#     if z_norm and (z_calib_dataset is not None):
#         calib_loader = torch.utils.data.DataLoader(
#             z_calib_dataset, batch_size=min(8, len(z_calib_dataset)),
#             shuffle=False, num_workers=0, drop_last=False
#         )
#         scores_c = []
#         with torch.no_grad():
#             for calib_img, _ in calib_loader:
#                 calib_img = calib_img.to(device)
#                 amap_c = _tta_merge(calib_img, do_flip=flip_tta, do_rot=do_rot_calib, scales=scales)

#                 # 统一 resize
#                 if resize_mask is not None:
#                     amap_c = F.interpolate(amap_c, size=resize_mask, mode='bilinear', align_corners=False)

#                 # 像素阈值校准（可选）
#                 if thr_tensor is not None:
#                     if thr_mode == 'sub':
#                         amap_c = torch.clamp(amap_c - thr_tensor, min=0.0)
#                     elif thr_mode == 'bin':
#                         amap_c = (amap_c > thr_tensor).float()

#                 # 平滑 + 后处理（可选）
#                 amap_c = gaussian_kernel(amap_c)
#                 if postproc:
#                     amap_c = _morph_smooth(amap_c, k=post_k, iters=post_iters)

#                 # 单通道规整
#                 if amap_c.shape[1] > 1:
#                     amap_c = amap_c.amax(dim=1, keepdim=True)

#                 # 图像级分数
#                 if aggregator is not None:
#                     img_scores_c = aggregator(
#                         hm=amap_c, enc_patch_tokens=None, H=None, W=None,
#                         k_ratio=topk_ratio, p=gem_p, fg_q=fg_q
#                     )
#                 else:
#                     amap_flat = amap_c.flatten(1)
#                     if max_ratio == 0:
#                         img_scores_c = torch.max(amap_flat, dim=1)[0]
#                     else:
#                         k = max(1, int(amap_flat.shape[1] * max_ratio))
#                         img_scores_c = torch.sort(amap_flat, dim=1, descending=True)[0][:, :k].mean(dim=1)

#                 scores_c.append(img_scores_c.detach().cpu())

#         scores_c = torch.cat(scores_c, dim=0).float()
#         mu = scores_c.mean().item()
#         sigma = scores_c.std(unbiased=False).item() + 1e-6

#     # ---------------- 正式评测 ----------------
#     with torch.no_grad():
#         for img, gt, label, img_path in dataloader:
#             img = img.to(device)

#             # TTA 融合
#             do_rot_eval = _should_rotate(_class_, rot_tta)
#             anomaly_map = _tta_merge(img, do_flip=flip_tta, do_rot=do_rot_eval, scales=scales)

#             # 统一 resize
#             if resize_mask is not None:
#                 anomaly_map = F.interpolate(anomaly_map, size=resize_mask, mode='bilinear', align_corners=False)
#                 gt = F.interpolate(gt, size=resize_mask, mode='nearest')
            
#             # >>>>>>>>>>>>>>>>>>>>>>> KNN MemoryBank 融合（可选） <<<<<<<<<<<<<<<<<<<<<<
#             if use_bank and (bank_encoder is not None) and (bank_feats is not None) and (bank_grid is not None):
#                 from rcm.matcher import rcm_score_batch
                
#                 # 计算 KNN 距离热图并上采样到 anomaly_map 的 H,W
#                 up_hw = anomaly_map.shape[-2:]
#                 knn = rcm_score_batch(
#                     bank_encoder,        # callable
#                     img,                 # [B,3,H,W]
#                     bank_feats,          # [M,C]
#                     bank_grid,           # (h,w)
#                     agg='quantile', agg_q=0.10,
#                     upsample_hw=up_hw
#                 )
#                 knn_map = knn['pix_map']  # [B,1,H,W]

#                 # 归一化到[0,1]（逐图），避免某类图强度偏移影响融合
#                 kmin = knn_map.amin(dim=(-1,-2), keepdim=True)
#                 kmax = knn_map.amax(dim=(-1,-2), keepdim=True)
#                 knn_map = (knn_map - kmin) / (kmax - kmin + 1e-6)

#                 # 凸组合：以 MSS 为主、KNN 补敏感
#                 anomaly_map = (1.0 - float(bank_alpha)) * anomaly_map + float(bank_alpha) * knn_map
#             # <<<<<<<<<<<<<<<<<<<<<< KNN MemoryBank 融合（可选） <<<<<<<<<<<<<<<<<<<<<<

#             # 像素阈值校准（可选）
#             if thr_tensor is not None:
#                 if thr_mode == 'sub':
#                     anomaly_map = torch.clamp(anomaly_map - thr_tensor, min=0.0)
#                 elif thr_mode == 'bin':
#                     anomaly_map = (anomaly_map > thr_tensor).float()

#             # 平滑 + 后处理（可选）
#             anomaly_map = gaussian_kernel(anomaly_map)
#             if postproc:
#                 anomaly_map = _morph_smooth(anomaly_map, k=post_k, iters=post_iters)

#             # 单通道规整
#             if anomaly_map.shape[1] > 1:
#                 anomaly_map = anomaly_map.amax(dim=1, keepdim=True)

#             # GT 处理
#             gt = gt.bool()
#             if gt.shape[1] > 1:
#                 gt = torch.max(gt, dim=1, keepdim=True)[0]

#             # 收集像素级
#             gt_list_px.append(gt)
#             pr_list_px.append(anomaly_map)

#             # 图像级分数
#             if aggregator is not None:
#                 sp_score = aggregator(
#                     hm=anomaly_map, enc_patch_tokens=None, H=None, W=None,
#                     k_ratio=topk_ratio, p=gem_p, fg_q=fg_q
#                 )
#             else:
#                 amap_flat = anomaly_map.flatten(1)
#                 if max_ratio == 0:
#                     sp_score = torch.max(amap_flat, dim=1)[0]
#                 else:
#                     k = max(1, int(amap_flat.shape[1] * max_ratio))
#                     sp_score = torch.sort(amap_flat, dim=1, descending=True)[0][:, :k].mean(dim=1)

#             # z-norm（图像级）
#             if (mu is not None) and (sigma is not None):
#                 sp_score = (sp_score - mu) / sigma

#             gt_list_sp.append(label)
#             pr_list_sp.append(sp_score)

#         # 计算指标
#         gt_list_px = torch.cat(gt_list_px, dim=0)[:, 0].cpu().numpy()
#         pr_list_px = torch.cat(pr_list_px, dim=0)[:, 0].cpu().numpy()
#         gt_list_sp = torch.cat(gt_list_sp).flatten().cpu().numpy()
#         pr_list_sp = torch.cat(pr_list_sp).flatten().cpu().numpy()

#         aupro_px = compute_pro(gt_list_px, pr_list_px)
#         gt_list_px, pr_list_px = gt_list_px.ravel(), pr_list_px.ravel()
#         auroc_px = roc_auc_score(gt_list_px, pr_list_px)
#         auroc_sp = roc_auc_score(gt_list_sp, pr_list_sp)
#         ap_px = average_precision_score(gt_list_px, pr_list_px)
#         ap_sp = average_precision_score(gt_list_sp, pr_list_sp)
#         f1_sp = f1_score_max(gt_list_sp, pr_list_sp)
#         f1_px = f1_score_max(gt_list_px, pr_list_px)

#     return [auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px]

# （已按方案B做“对齐到patch倍数”的TTA改造 + 兼容像素级指标开关）
def evaluation_batch(
    model,
    dataloader,
    device,
    _class_=None,
    max_ratio=0,
    resize_mask=None,

    # 你已有的参数
    tta_hflip: bool = False,
    per_class_thr: Optional[float] = None,
    thr_mode: str = "sub",
    aggregator=None,
    flip_tta: Optional[bool] = None,
    z_norm: bool = False,
    z_calib_dataset=None,
    postproc: bool = False,
    topk_ratio: float = 0.02,
    gem_p: float = 6.0,
    fg_q: int = 70,
    post_k: int = 3,
    post_iters: int = 1,

    # —— KNN 记忆库融合（可选）——
    use_bank: bool = False,
    bank_encoder=None,                 # callable(images)-> {'tokens':...}或{'patch':...}
    bank_feats: 'torch.Tensor | None' = None,   # [M,C]
    bank_grid: 'tuple | None' = None,           # (h,w)，来自MemoryBank
    bank_alpha: float = 0.30,                   # 融合权重
    bank_topk: int = 1,                         # KNN的k

    # ------- 旋转与多尺度 TTA -------
    rot_tta: 'str | None' = None,   # 'none' | 'all' | 'textures' | 'cls1,cls2'
    ms_tta: 'str | None' = None,    # 如 "0.75,1.0"（1.0 会被忽略，因为原图已包含）

    # ------- 新增：像素级指标开关（默认保持旧行为：开启）-------
    px_metrics: bool = True,

    # ------- 新增：sample-level 指标（多视角聚合）-------
    sample_metrics: bool = False,

    # ------- 新增：可选的分组权重（如方差驱动的自适应加权）-------
    group_weights: 'list[torch.Tensor] | None' = None,
    
    # ------- 融合策略参数 -------
    fusion_mode: str = 'mean',
    fusion_confidence: str = 'topk_mean',
    fusion_topk: float = 0.02,
    fusion_temp: float = 0.07,
    fusion_gate=None,
):
    if flip_tta is None:
        flip_tta = tta_hflip

    model.eval()
    gt_list_px, pr_list_px = [], []
    gt_list_sp, pr_list_sp = [], []
    group_ids_all = [] if sample_metrics else None
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    # ============== 读取 patch_size（用于TTA对齐） ==============
    patch_h = patch_w = 14
    pe = getattr(model, 'encoder', None)
    pe = getattr(pe, 'patch_embed', None) if pe is not None else None
    ps = getattr(pe, 'patch_size', None) if pe is not None else None
    if isinstance(ps, (tuple, list)) and len(ps) == 2:
        patch_h, patch_w = int(ps[0]), int(ps[1])
    elif isinstance(ps, int):
        patch_h = patch_w = int(ps)

    # 像素阈值 → 张量
    thr_tensor = None
    if per_class_thr is not None:
        thr_tensor = torch.tensor(float(per_class_thr), device=device).view(1, 1, 1, 1)

    # ---------- 小工具：单次前向得到热图 ----------
    @torch.no_grad()
    def _fwd_heatmap(x):
        en, de = model(x)
        # [MODIFIED] 添加融合模式参数
        amap, _ = cal_anomaly_maps(
            en, de, x.shape[-1], 
            weights=group_weights,
            fusion_mode=fusion_mode,
            fusion_confidence=fusion_confidence,
            fusion_topk=fusion_topk,
            fusion_temp=fusion_temp,
            fusion_gate=fusion_gate,
        )
        return amap


    # ============== 按比例缩放→对齐到patch倍数 ==============
    def _resize_to_patch_multiple(x, scale: float, ph: int, pw: int):
        # x: [B, C, H, W]
        H, W = x.shape[-2], x.shape[-1]
        Ht = int(round(H * scale))
        Wt = int(round(W * scale))
        # 对齐到 patch 的整数倍（四舍五入到最近倍数）
        Ht = max(ph, int(round(Ht / ph)) * ph)
        Wt = max(pw, int(round(Wt / pw)) * pw)
        return F.interpolate(x, size=(Ht, Wt), mode='bilinear', align_corners=False)

    # ---------- 小工具：对一批图像做 TTA（独立合并，不做笛卡尔乘积） ----------
    def _tta_merge(img, do_flip: bool, do_rot: bool, scales: 'list[float]'):
        H, W = img.shape[-2], img.shape[-1]
        maps = []

        # 原图
        amap = _fwd_heatmap(img)
        maps.append(amap)

        # 水平翻转（逐像素max融合）
        if do_flip:
            x = torch.flip(img, dims=[-1])
            a = _fwd_heatmap(x)
            a = torch.flip(a, dims=[-1])
            maps.append(a)

        # 旋转 90/180/270（只对白名单类启用）
        if do_rot:
            for k in (1, 2, 3):
                x = torch.rot90(img, k=k, dims=[-2, -1])
                a = _fwd_heatmap(x)
                a = torch.rot90(a, k=4 - k, dims=[-2, -1])
                maps.append(a)

        # 多尺度（例如 0.75×/1.25×），回到原分辨率再融合
        for s in scales:
            if abs(float(s) - 1.0) < 1e-6:
                continue
            x_s = _resize_to_patch_multiple(img, float(s), patch_h, patch_w)
            a = _fwd_heatmap(x_s)
            a = F.interpolate(a, size=(H, W), mode='bilinear', align_corners=False)
            maps.append(a)

        # 逐像素 max 融合
        out = maps[0]
        for m in maps[1:]:
            out = torch.maximum(out, m)
        return out  # [B,?,H,W]

    # ---------------- z-norm μ/σ 统计（如启用） ----------------
    mu = sigma = None
    do_rot_calib = _should_rotate(_class_, rot_tta)
    scales = _parse_scales(ms_tta)

    if z_norm and (z_calib_dataset is not None):
        calib_loader = torch.utils.data.DataLoader(
            z_calib_dataset, batch_size=min(8, len(z_calib_dataset)),
            shuffle=False, num_workers=0, drop_last=False
        )
        scores_c = []
        with torch.no_grad():
            for calib_img, _ in calib_loader:
                calib_img = calib_img.to(device)
                amap_c = _tta_merge(calib_img, do_flip=flip_tta, do_rot=do_rot_calib, scales=scales)

                if resize_mask is not None:
                    amap_c = F.interpolate(amap_c, size=resize_mask, mode='bilinear', align_corners=False)

                if thr_tensor is not None:
                    if thr_mode == 'sub':
                        amap_c = torch.clamp(amap_c - thr_tensor, min=0.0)
                    elif thr_mode == 'bin':
                        amap_c = (amap_c > thr_tensor).float()

                amap_c = gaussian_kernel(amap_c)
                if postproc:
                    amap_c = _morph_smooth(amap_c, k=post_k, iters=post_iters)

                if amap_c.shape[1] > 1:
                    amap_c = amap_c.amax(dim=1, keepdim=True)

                if aggregator is not None:
                    img_scores_c = aggregator(
                        hm=amap_c, enc_patch_tokens=None, H=None, W=None,
                        k_ratio=topk_ratio, p=gem_p, fg_q=fg_q
                    )
                else:
                    amap_flat = amap_c.flatten(1)
                    if max_ratio == 0:
                        img_scores_c = torch.max(amap_flat, dim=1)[0]
                    else:
                        k = max(1, int(amap_flat.shape[1] * max_ratio))
                        img_scores_c = torch.sort(amap_flat, dim=1, descending=True)[0][:, :k].mean(dim=1)

                scores_c.append(img_scores_c.detach().cpu())

        scores_c = torch.cat(scores_c, dim=0).float()
        mu = scores_c.mean().item()
        sigma = scores_c.std(unbiased=False).item() + 1e-6

    def _to_group_list(group_id, bsz):
        if group_id is None:
            return [None] * bsz
        if isinstance(group_id, (list, tuple)):
            out = [str(x) for x in group_id]
        elif isinstance(group_id, torch.Tensor):
            if group_id.numel() == 1:
                out = [str(group_id.item())]
            else:
                out = [str(x) for x in group_id.flatten().tolist()]
        else:
            out = [str(group_id)]
        if len(out) != bsz:
            out = [out[0]] * bsz if out else [None] * bsz
        return out

    # ---------------- 正式评测 ----------------
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, (list, tuple)) and len(batch) == 5:
                img, gt, label, img_path, group_id = batch
            elif isinstance(batch, (list, tuple)) and len(batch) == 4:
                img, gt, label, img_path = batch
                group_id = None
            else:
                raise ValueError("evaluation_batch expects 4 or 5 fields from dataloader.")
            img = img.to(device)

            # TTA 融合
            do_rot_eval = _should_rotate(_class_, rot_tta)
            anomaly_map = _tta_merge(img, do_flip=flip_tta, do_rot=do_rot_eval, scales=scales)

            # 统一 resize（像素管线需要）
            if resize_mask is not None:
                anomaly_map = F.interpolate(anomaly_map, size=resize_mask, mode='bilinear', align_corners=False)
                gt = F.interpolate(gt, size=resize_mask, mode='nearest')

            # >>> KNN MemoryBank 融合（可选） >>>
            if use_bank and (bank_encoder is not None) and (bank_feats is not None) and (bank_grid is not None):
                from rcm.matcher import rcm_score_batch
                up_hw = anomaly_map.shape[-2:]
                knn = rcm_score_batch(
                    bank_encoder, img, bank_feats, bank_grid,
                    agg='quantile', agg_q=0.10, upsample_hw=up_hw
                )
                knn_map = knn['pix_map']  # [B,1,H,W]
                kmin = knn_map.amin(dim=(-1,-2), keepdim=True)
                kmax = knn_map.amax(dim=(-1,-2), keepdim=True)
                knn_map = (knn_map - kmin) / (kmax - kmin + 1e-6)
                anomaly_map = (1.0 - float(bank_alpha)) * anomaly_map + float(bank_alpha) * knn_map
            # <<< KNN MemoryBank 融合（可选） <<<

            # 像素阈值校准（可选）
            if thr_tensor is not None:
                if thr_mode == 'sub':
                    anomaly_map = torch.clamp(anomaly_map - thr_tensor, min=0.0)
                elif thr_mode == 'bin':
                    anomaly_map = (anomaly_map > thr_tensor).float()

            # 平滑 + 后处理（可选）
            anomaly_map = gaussian_kernel(anomaly_map)
            if postproc:
                anomaly_map = _morph_smooth(anomaly_map, k=post_k, iters=post_iters)

            # 单通道规整
            if anomaly_map.shape[1] > 1:
                anomaly_map = anomaly_map.amax(dim=1, keepdim=True)

            # GT 处理（允许无掩码：数据集已给零张量，这里仍做布尔化与单通道）
            gt = gt.bool()
            if gt.shape[1] > 1:
                gt = torch.max(gt, dim=1, keepdim=True)[0]

            # 收集像素级（稍后根据开关决定是否参与指标计算）
            gt_list_px.append(gt)
            pr_list_px.append(anomaly_map)

            # 图像级分数
            if aggregator is not None:
                sp_score = aggregator(
                    hm=anomaly_map, enc_patch_tokens=None, H=None, W=None,
                    k_ratio=topk_ratio, p=gem_p, fg_q=fg_q
                )
            else:
                amap_flat = anomaly_map.flatten(1)
                if max_ratio == 0:
                    sp_score = torch.max(amap_flat, dim=1)[0]
                else:
                    k = max(1, int(amap_flat.shape[1] * max_ratio))
                    sp_score = torch.sort(amap_flat, dim=1, descending=True)[0][:, :k].mean(dim=1)

            # z-norm（图像级）
            if (mu is not None) and (sigma is not None):
                sp_score = (sp_score - mu) / sigma

            gt_list_sp.append(label)
            pr_list_sp.append(sp_score)
            if sample_metrics:
                group_ids_all.extend(_to_group_list(group_id, img.shape[0]))

        # ===== 汇总与指标 =====
        gt_list_sp = torch.cat(gt_list_sp).flatten().cpu().numpy()
        pr_list_sp = torch.cat(pr_list_sp).flatten().cpu().numpy()
        auroc_sp = roc_auc_score(gt_list_sp, pr_list_sp)
        ap_sp    = average_precision_score(gt_list_sp, pr_list_sp)
        f1_sp    = f1_score_max(gt_list_sp, pr_list_sp)

        # --- 像素级指标（仅在明确启用 & 具备必要条件时计算）---
        # 条件：px_metrics=True 且 resize_mask!=None 且 max_ratio>0（代表像素管线是打开的）
        enable_px = bool(px_metrics) and (resize_mask is not None) and (max_ratio and max_ratio > 0)

        if enable_px:
            gt_list_px_np = torch.cat(gt_list_px, dim=0)[:, 0].float().cpu().numpy()
            pr_list_px_np = torch.cat(pr_list_px, dim=0)[:, 0].float().cpu().numpy()

            # 二值化保证 {0,1}
            gt_list_px_np = (gt_list_px_np >= 0.5).astype(np.uint8)

            # 若没有任何正像素（例如 Real-IAD 无掩码），则跳过像素级指标
            if gt_list_px_np.sum() == 0 or set(np.unique(gt_list_px_np)) == {0}:
                auroc_px = float('nan'); ap_px = float('nan'); f1_px = float('nan'); aupro_px = float('nan')
            else:
                aupro_px = compute_pro(gt_list_px_np, pr_list_px_np)
                gt_flat = gt_list_px_np.ravel()
                pr_flat = pr_list_px_np.ravel()
                auroc_px = roc_auc_score(gt_flat, pr_flat)
                ap_px    = average_precision_score(gt_flat, pr_flat)
                f1_px    = f1_score_max(gt_flat, pr_flat)
        else:
            auroc_px = float('nan'); ap_px = float('nan'); f1_px = float('nan'); aupro_px = float('nan')

        if sample_metrics:
            auroc_sa = float('nan'); ap_sa = float('nan'); f1_sa = float('nan')
            if group_ids_all is not None and len(group_ids_all) == len(gt_list_sp):
                group_scores = {}
                group_labels = {}
                for i, (gid, score, lab) in enumerate(zip(group_ids_all, pr_list_sp, gt_list_sp)):
                    key = gid if gid not in (None, '') else f"__img_{i}__"
                    if key in group_scores:
                        group_scores[key] = max(group_scores[key], float(score))
                        group_labels[key] = max(group_labels[key], int(lab))
                    else:
                        group_scores[key] = float(score)
                        group_labels[key] = int(lab)

                keys = list(group_labels.keys())
                labels_sa = np.array([group_labels[k] for k in keys], dtype=np.int32)
                scores_sa = np.array([group_scores[k] for k in keys], dtype=np.float32)
                if np.unique(labels_sa).size > 1:
                    auroc_sa = roc_auc_score(labels_sa, scores_sa)
                    ap_sa = average_precision_score(labels_sa, scores_sa)
                    f1_sa = f1_score_max(labels_sa, scores_sa)
            return [auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px, auroc_sa, ap_sa, f1_sa]

    return [auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px]


def evaluation_batch_loco(model, dataloader, device, _class_=None, max_ratio=0):
    model.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    defect_type_list = []
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    with torch.no_grad():
        for img, gt, label, path, defect_type, size in dataloader:
            img = img.to(device)

            output = model(img)
            en, de = output[0], output[1]

            anomaly_map, _ = cal_anomaly_maps(en, de, img.shape[-1])
            anomaly_map = gaussian_kernel(anomaly_map)

            gt = gt.bool()

            gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
            pr_list_px.extend(anomaly_map.cpu().numpy().ravel())
            gt_list_sp.extend(label.cpu().numpy().astype(int))

            if max_ratio == 0:
                sp_score = torch.max(anomaly_map.flatten(1), dim=1)[0].cpu().numpy()
            else:
                anomaly_map = anomaly_map.flatten(1)
                sp_score = torch.sort(anomaly_map, dim=1, descending=True)[0][:, :int(anomaly_map.shape[1] * max_ratio)]
                sp_score = sp_score.mean(dim=1).cpu().numpy()
            pr_list_sp.extend(sp_score)
            defect_type_list.extend(defect_type)

        auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 4)
        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 4)
        ap_px = round(average_precision_score(gt_list_px, pr_list_px), 4)
        ap_sp = round(average_precision_score(gt_list_sp, pr_list_sp), 4)

        defect_type_list = np.array(defect_type_list)
        auroc_logic = roc_auc_score(
            np.array(gt_list_sp)[np.logical_or(defect_type_list == 'good', defect_type_list == 'logical_anomalies')],
            np.array(pr_list_sp)[np.logical_or(defect_type_list == 'good', defect_type_list == 'logical_anomalies')])
        auroc_struct = roc_auc_score(
            np.array(gt_list_sp)[np.logical_or(defect_type_list == 'good', defect_type_list == 'structural_anomalies')],
            np.array(pr_list_sp)[np.logical_or(defect_type_list == 'good', defect_type_list == 'structural_anomalies')])
        auroc_both = (auroc_logic + auroc_struct) / 2

    return auroc_sp, auroc_logic, auroc_struct, auroc_both


def evaluation_uniad(model, dataloader, device, _class_=None, reg_calib=False, max_ratio=0):
    model.eval()
    gt_list_px = []
    pr_list_px = []
    gt_list_sp = []
    pr_list_sp = []
    aupro_list = []
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    with torch.no_grad():
        for img, gt, label, _ in dataloader:
            img = img.to(device)
            if reg_calib:
                en, de, reg = model({'image': img})
            else:
                en, de = model({'image': img})

            anomaly_map = torch.mean(F.mse_loss(de, en, reduction='none'), dim=1, keepdim=True)
            anomaly_map = F.interpolate(anomaly_map, size=(img.shape[-1], img.shape[-1]), mode='bilinear',
                                        align_corners=False)

            if reg_calib:
                if reg.shape[1] == 2:
                    reg_mean = reg[:, 0].view(-1, 1, 1, 1)
                    reg_max = reg[:, 1].view(-1, 1, 1, 1)
                    anomaly_map = (anomaly_map - reg_mean) / (reg_max - reg_mean)
                    # anomaly_map = anomaly_map - reg_max

                else:
                    reg = F.interpolate(reg, size=img.shape[-1], mode='bilinear', align_corners=True)
                    anomaly_map = anomaly_map - reg

            anomaly_map = gaussian_kernel(anomaly_map)

            gt = gt.bool()

            gt_list_px.extend(gt.cpu().numpy().astype(int).ravel())
            pr_list_px.extend(anomaly_map.cpu().numpy().ravel())
            gt_list_sp.extend(label.cpu().numpy().astype(int))

            if max_ratio == 0:
                sp_score = torch.max(anomaly_map.flatten(1), dim=1)[0].cpu().numpy()
            else:
                anomaly_map = anomaly_map.flatten(1)
                sp_score = torch.sort(anomaly_map, dim=1, descending=True)[0][:, :int(anomaly_map.shape[1] * max_ratio)]
                sp_score = sp_score.mean(dim=1).cpu().numpy()
            pr_list_sp.extend(sp_score)

        auroc_px = round(roc_auc_score(gt_list_px, pr_list_px), 4)
        auroc_sp = round(roc_auc_score(gt_list_sp, pr_list_sp), 4)
        ap_px = round(average_precision_score(gt_list_px, pr_list_px), 4)
        ap_sp = round(average_precision_score(gt_list_sp, pr_list_sp), 4)

    return auroc_px, auroc_sp, ap_px, ap_sp, [gt_list_px, pr_list_px, gt_list_sp, pr_list_sp]


def visualize(model, dataloader, device, _class_='None', save_name='save'):
    model.eval()
    save_dir = os.path.join('./visualize', save_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    gaussian_kernel = get_gaussian_kernel(kernel_size=5, sigma=4).to(device)

    with torch.no_grad():
        for img, gt, label, img_path in dataloader:
            img = img.to(device)
            output = model(img)
            en, de = output[0], output[1]
            anomaly_map, _ = cal_anomaly_maps(en, de, img.shape[-1])
            anomaly_map = gaussian_kernel(anomaly_map)

            for i in range(0, anomaly_map.shape[0], 8):
                heatmap = min_max_norm(anomaly_map[i, 0].cpu().numpy())
                heatmap = cvt2heatmap(heatmap * 255)
                im = img[i].permute(1, 2, 0).cpu().numpy()
                im = im * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
                im = (im * 255).astype('uint8')
                im = im[:, :, ::-1]
                hm_on_img = show_cam_on_image(im, heatmap)
                mask = (gt[i][0].numpy() * 255).astype('uint8')
                save_dir_class = os.path.join(save_dir, str(_class_))
                if not os.path.exists(save_dir_class):
                    os.mkdir(save_dir_class)
                name = img_path[i].split('/')[-2] + '_' + img_path[i].split('/')[-1].replace('.png', '')
                cv2.imwrite(save_dir_class + '/' + name + '_img.png', im)
                cv2.imwrite(save_dir_class + '/' + name + '_cam.png', hm_on_img)
                cv2.imwrite(save_dir_class + '/' + name + '_gt.png', mask)

    return


def save_feature(model, dataloader, device, _class_='None', save_name='save'):
    model.eval()
    save_dir = os.path.join('./feature', save_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    with torch.no_grad():
        for img, gt, label, img_path in dataloader:
            img = img.to(device)
            en, de = model(img)

            en_abnorm_list = []
            en_normal_list = []
            de_abnorm_list = []
            de_normal_list = []

            for i in range(3):
                en_feat = en[0 + i]
                de_feat = de[0 + i]

                gt_resize = F.interpolate(gt, size=en_feat.shape[2], mode='bilinear') > 0

                en_abnorm = en_feat.permute(0, 2, 3, 1)[gt_resize.permute(0, 2, 3, 1)[:, :, :, 0]]
                en_normal = en_feat.permute(0, 2, 3, 1)[gt_resize.permute(0, 2, 3, 1)[:, :, :, 0] == 0]

                de_abnorm = de_feat.permute(0, 2, 3, 1)[gt_resize.permute(0, 2, 3, 1)[:, :, :, 0]]
                de_normal = de_feat.permute(0, 2, 3, 1)[gt_resize.permute(0, 2, 3, 1)[:, :, :, 0] == 0]

                en_abnorm_list.append(F.normalize(en_abnorm, dim=1).cpu().numpy())
                en_normal_list.append(F.normalize(en_normal, dim=1).cpu().numpy())
                de_abnorm_list.append(F.normalize(de_abnorm, dim=1).cpu().numpy())
                de_normal_list.append(F.normalize(de_normal, dim=1).cpu().numpy())

            save_dir_class = os.path.join(save_dir, str(_class_))
            if not os.path.exists(save_dir_class):
                os.mkdir(save_dir_class)
            name = img_path[0].split('/')[-2] + '_' + img_path[0].split('/')[-1].replace('.png', '')

            saved_dict = {'en_abnorm_list': en_abnorm_list, 'en_normal_list': en_normal_list,
                          'de_abnorm_list': de_abnorm_list, 'de_normal_list': de_normal_list}

            with open(save_dir_class + '/' + name + '.pkl', 'wb') as f:
                pickle.dump(saved_dict, f)

    return


def visualize_noseg(model, dataloader, device, _class_='None', save_name='save'):
    model.eval()
    save_dir = os.path.join('./visualize', save_name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    with torch.no_grad():
        for img, label, img_path in dataloader:
            img = img.to(device)
            en, de = model(img)

            anomaly_map, _ = cal_anomaly_map(en, de, img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)

            heatmap = min_max_norm(anomaly_map)
            heatmap = cvt2heatmap(heatmap * 255)
            img = img.permute(0, 2, 3, 1).cpu().numpy()[0]
            img = img * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])
            img = (img * 255).astype('uint8')
            hm_on_img = show_cam_on_image(img, heatmap)

            save_dir_class = os.path.join(save_dir, str(_class_))
            if not os.path.exists(save_dir_class):
                os.mkdir(save_dir_class)
            name = img_path[0].split('/')[-2] + '_' + img_path[0].split('/')[-1].replace('.png', '')
            cv2.imwrite(save_dir_class + '/' + name + '_seg.png', heatmap)
            cv2.imwrite(save_dir_class + '/' + name + '_cam.png', hm_on_img)

    return


def visualize_loco(model, dataloader, device, _class_='None', save_name='save'):
    model.eval()
    save_dir = os.path.join('./visualize', save_name)
    with torch.no_grad():
        for img, gt, label, img_path, defect_type, size in dataloader:
            img = img.to(device)
            en, de = model(img)

            anomaly_map, _ = cal_anomaly_map(en, de, img.shape[-1], amap_mode='a')
            anomaly_map = gaussian_filter(anomaly_map, sigma=4)
            anomaly_map = cv2.resize(anomaly_map, dsize=(size[0].item(), size[1].item()),
                                     interpolation=cv2.INTER_NEAREST)

            save_dir_class = os.path.join(save_dir, str(_class_), 'test', defect_type[0])
            if not os.path.exists(save_dir_class):
                os.makedirs(save_dir_class)
            name = img_path[0].split('/')[-1].replace('.png', '')
            cv2.imwrite(save_dir_class + '/' + name + '.tiff', anomaly_map)
    return


def compute_pro(masks: ndarray, amaps: ndarray, num_th: int = 200) -> None:
    """Compute the area under the curve of per-region overlaping (PRO) and 0 to 0.3 FPR
    Args:
        category (str): Category of product
        masks (ndarray): All binary masks in test. masks.shape -> (num_test_data, h, w)
        amaps (ndarray): All anomaly maps in test. amaps.shape -> (num_test_data, h, w)
        num_th (int, optional): Number of thresholds
    """

    assert isinstance(amaps, ndarray), "type(amaps) must be ndarray"
    assert isinstance(masks, ndarray), "type(masks) must be ndarray"
    assert amaps.ndim == 3, "amaps.ndim must be 3 (num_test_data, h, w)"
    assert masks.ndim == 3, "masks.ndim must be 3 (num_test_data, h, w)"
    assert amaps.shape == masks.shape, "amaps.shape and masks.shape must be same"
    assert set(masks.flatten()) == {0, 1}, "set(masks.flatten()) must be {0, 1}"
    assert isinstance(num_th, int), "type(num_th) must be int"

    df = pd.DataFrame([], columns=["pro", "fpr", "threshold"])
    binary_amaps = np.zeros_like(amaps, dtype=bool)  # 或 dtype=np.bool_

    min_th = amaps.min()
    max_th = amaps.max()
    delta = (max_th - min_th) / num_th

    for th in np.arange(min_th, max_th, delta):
        binary_amaps[amaps <= th] = 0
        binary_amaps[amaps > th] = 1

        pros = []
        for binary_amap, mask in zip(binary_amaps, masks):
            for region in measure.regionprops(measure.label(mask)):
                axes0_ids = region.coords[:, 0]
                axes1_ids = region.coords[:, 1]
                tp_pixels = binary_amap[axes0_ids, axes1_ids].sum()
                pros.append(tp_pixels / region.area)

        inverse_masks = 1 - masks
        fp_pixels = np.logical_and(inverse_masks, binary_amaps).sum()
        fpr = fp_pixels / inverse_masks.sum()

        # df = df.append({"pro": mean(pros), "fpr": fpr, "threshold": th}, ignore_index=True)
        df.loc[len(df)] = {"pro": mean(pros), "fpr": fpr, "threshold": th}

    # Normalize FPR from 0 ~ 1 to 0 ~ 0.3
    df = df[df["fpr"] < 0.3]
    df["fpr"] = df["fpr"] / df["fpr"].max()

    pro_auc = auc(df["fpr"], df["pro"])
    return pro_auc


def get_gaussian_kernel(kernel_size=3, sigma=2, channels=1):
    # Create a x, y coordinate grid of shape (kernel_size, kernel_size, 2)
    x_coord = torch.arange(kernel_size)
    x_grid = x_coord.repeat(kernel_size).view(kernel_size, kernel_size)
    y_grid = x_grid.t()
    xy_grid = torch.stack([x_grid, y_grid], dim=-1).float()

    mean = (kernel_size - 1) / 2.
    variance = sigma ** 2.

    # Calculate the 2-dimensional gaussian kernel which is
    # the product of two gaussian distributions for two different
    # variables (in this case called x and y)
    gaussian_kernel = (1. / (2. * math.pi * variance)) * \
                      torch.exp(
                          -torch.sum((xy_grid - mean) ** 2., dim=-1) / \
                          (2 * variance)
                      )

    # Make sure sum of values in gaussian kernel equals 1.
    gaussian_kernel = gaussian_kernel / torch.sum(gaussian_kernel)

    # Reshape to 2d depthwise convolutional weight
    gaussian_kernel = gaussian_kernel.view(1, 1, kernel_size, kernel_size)
    gaussian_kernel = gaussian_kernel.repeat(channels, 1, 1, 1)

    gaussian_filter = torch.nn.Conv2d(in_channels=channels, out_channels=channels, kernel_size=kernel_size,
                                      groups=channels,
                                      bias=False, padding=kernel_size // 2)

    gaussian_filter.weight.data = gaussian_kernel
    gaussian_filter.weight.requires_grad = False

    return gaussian_filter


class FeatureJitter(torch.nn.Module):
    def __init__(self, scale=1., p=0.25) -> None:
        super(FeatureJitter, self).__init__()
        self.scale = scale
        self.p = p

    def add_jitter(self, feature):
        if self.scale > 0:
            B, C, H, W = feature.shape
            feature_norms = feature.norm(dim=1).unsqueeze(1) / C  # B*1*H*W
            jitter = torch.randn((B, C, H, W), device=feature.device)
            jitter = F.normalize(jitter, dim=1)
            jitter = jitter * feature_norms * self.scale
            mask = torch.rand((B, 1, H, W), device=feature.device) < self.p
            feature = feature + jitter * mask
        return feature

    def forward(self, x):
        if self.training:
            x = self.add_jitter(x)
        return x


def replace_layers(model, old, new):
    for n, module in model.named_children():
        if len(list(module.children())) > 0:
            ## compound module, go inside it
            replace_layers(module, old, new)

        if isinstance(module, old):
            ## simple module
            setattr(model, n, new)


from torch.optim.lr_scheduler import _LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau


class WarmCosineScheduler(_LRScheduler):

    def __init__(self, optimizer, base_value, final_value, total_iters, warmup_iters=0, start_warmup_value=0, ):
        self.final_value = final_value
        self.total_iters = total_iters
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

        iters = np.arange(total_iters - warmup_iters)
        schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
        self.schedule = np.concatenate((warmup_schedule, schedule))

        super(WarmCosineScheduler, self).__init__(optimizer)

    def get_lr(self):
        if self.last_epoch >= self.total_iters:
            return [self.final_value for base_lr in self.base_lrs]
        else:
            return [self.schedule[self.last_epoch] for base_lr in self.base_lrs]
import torch
import torch.nn.functional as F

def per_image_calib(amap):
    # amap: [B,1,H,W] -> z-score + 上尾截断（鲁棒）
    B = amap.size(0)
    x  = amap.flatten(2)                                # [B,1,HW]
    mu = x.mean(-1, keepdim=True)
    sd = x.std(-1, keepdim=True).clamp_min(1e-6)
    x  = (x - mu) / sd
    q  = x.quantile(0.995, dim=-1, keepdim=True)
    x  = x.clamp_max(q)
    return x.view_as(amap)

def tta_rot_ens(forward_once, img, modes=('rot0','rot90','rot180','rot270')):
    outs = []
    for m in modes:
        if   m=='rot0':   im = img
        elif m=='rot90':  im = img.rot90(1, dims=[-2,-1])
        elif m=='rot180': im = img.rot90(2, dims=[-2,-1])
        elif m=='rot270': im = img.rot90(3, dims=[-2,-1])
        amap = forward_once(im)                          # -> [B,1,H,W]
        if   m=='rot90':  amap = amap.rot90(3, dims=[-2,-1])
        elif m=='rot180': amap = amap.rot90(2, dims=[-2,-1])
        elif m=='rot270': amap = amap.rot90(1, dims=[-2,-1])
        outs.append(amap)
    outs = torch.stack(outs, dim=0)                     # [M,B,1,H,W]
    return outs.mean(0), outs.max(0).values

# utils/fast_guided_filter.py

@torch.no_grad()
def _box_filter(x, r):
    k = 2 * r + 1
    return F.avg_pool2d(x, kernel_size=k, stride=1, padding=r)

@torch.no_grad()
def fast_guided_filter_batch(guide, src, r=8, eps=1e-4, s=4):
    B, _, H, W = guide.shape
    h, w = max(2, H//s), max(2, W//s)
    I = F.interpolate(guide, (h,w), mode='bilinear', align_corners=False)
    p = F.interpolate(src,   (h,w), mode='bilinear', align_corners=False)

    r_sub = max(1, r//s)
    mean_I  = _box_filter(I, r_sub)
    mean_p  = _box_filter(p, r_sub)
    mean_II = _box_filter(I*I, r_sub)
    mean_Ip = _box_filter(I*p, r_sub)

    var_I  = mean_II - mean_I*mean_I
    cov_Ip = mean_Ip - mean_I*mean_p

    a = cov_Ip / (var_I + eps)
    b = mean_p - a*mean_I
    a = a.mean(1, True); b = b.mean(1, True)

    a = F.interpolate(a, (H,W), mode='bilinear', align_corners=False)
    b = F.interpolate(b, (H,W), mode='bilinear', align_corners=False)
    q = a * guide.mean(1, True) + b
    return q.clamp_min(0)
