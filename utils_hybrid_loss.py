# ========== [NEW] 混合损失函数 (Cosine + L2) ==========
# 用于支持灵活的损失函数切换，特别适配 Hybrid Decoder (ViT + Mamba)

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import global_cosine_hm_percent


def compute_hard_mining_p(
    it: int,
    total_iters: int,
    p_start: float,
    p_end: float,
    schedule: str = "cosine",
    warmup: int = 0,
    cycle_iters: int = 0,
):
    if total_iters <= 0:
        return float(max(0.0, min(0.999, p_end)))
    if schedule == "warmup":
        if warmup <= 0:
            return float(max(0.0, min(0.999, p_end)))
        p = min(p_end * it / max(1, warmup), p_end)
        return float(max(0.0, min(0.999, p)))

    if warmup > 0 and it < warmup:
        p = p_start * it / max(1, warmup)
        return float(max(0.0, min(0.999, p)))

    denom = max(1, total_iters - warmup)
    progress = (it - warmup) / denom
    progress = max(0.0, min(1.0, progress))

    if schedule == "linear":
        p = p_start + (p_end - p_start) * progress
    elif schedule == "cosine":
        p = p_end - (p_end - p_start) * (0.5 * (1.0 + math.cos(math.pi * progress)))
    elif schedule == "cycle":
        cycle = cycle_iters if cycle_iters > 0 else denom
        cycle = max(1, int(cycle))
        t = ((it - warmup) % cycle) / cycle
        if t < 0.5:
            p = p_start + (p_end - p_start) * (t * 2.0)
        else:
            p = p_end - (p_end - p_start) * ((t - 0.5) * 2.0)
    else:  # fixed / unknown
        p = p_end
    return float(max(0.0, min(0.999, p)))


def local_pointwise_cosine_loss(a, b):
    cos_loss = nn.CosineSimilarity()
    loss = 0.0
    for item in range(len(a)):
        a_ = a[item].detach()
        b_ = b[item]
        point_dist = 1 - cos_loss(a_, b_).unsqueeze(1)
        loss += point_dist.mean()
    return loss / len(a)


class LearnableGroupWeights(nn.Module):
    def __init__(self, num_groups: int, temperature: float = 1.0):
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(num_groups, dtype=torch.float32))
        self.temperature = float(temperature)

    def forward(self):
        temp = self.temperature if self.temperature > 1e-6 else 1e-6
        return torch.softmax(self.logits / temp, dim=0)

# ==========================================
# [NEW] 新增损失函数: SSIM, Gradient, Orthogonal
# ==========================================

def gaussian_window(size, sigma):
    coords = torch.arange(size, dtype=torch.float)
    coords -= size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g /= g.sum()
    return g.reshape(1, 1, 1, -1) * g.reshape(1, 1, -1, 1)

def ssim_loss_func(x, y, window_size=11, sigma=1.5):
    """
    计算单通道 SSIM 损失 (1 - SSIM)
    x, y: [B, C, H, W]
    """
    # 确保输入是 4D
    if x.ndim == 3:
        x = x.unsqueeze(1)
    if y.ndim == 3:
        y = y.unsqueeze(1)
        
    # 动态创建窗口 (放到同一设备)
    window = gaussian_window(window_size, sigma).to(x.device).type_as(x)
    # 扩展到通道数 (假设是 depthwise)
    channels = x.shape[1]
    window = window.expand(channels, 1, window_size, window_size)
    
    mu1 = F.conv2d(x, window, padding=window_size//2, groups=channels)
    mu2 = F.conv2d(y, window, padding=window_size//2, groups=channels)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(x * x, window, padding=window_size//2, groups=channels) - mu1_sq
    sigma2_sq = F.conv2d(y * y, window, padding=window_size//2, groups=channels) - mu2_sq
    sigma12 = F.conv2d(x * y, window, padding=window_size//2, groups=channels) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / \
               ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return 1 - ssim_map.mean()

def global_ssim_loss(a, b, window_size=11):
    """
    全局 SSIM 损失
    Args:
        a: encoder features (list)
        b: decoder features (list)
    """
    loss = 0
    for item in range(len(a)):
        feat_a = a[item]
        feat_b = b[item]
        
        # Handle [B, N, C] -> [B, C, H, W] if needed
        if feat_a.ndim == 3:
            B, N, C = feat_a.shape
            H = int(N ** 0.5)
            if H * H == N:
                feat_a = feat_a.permute(0, 2, 1).reshape(B, C, H, H)
                feat_b = feat_b.permute(0, 2, 1).reshape(B, C, H, H)
        
        loss += ssim_loss_func(feat_a, feat_b, window_size=window_size)
        
    return loss / len(a)


def gradient_loss_func(x, y):
    """
    计算梯度损失 (Sobel)
    x, y: [B, C, H, W]
    """
    # Sobel 核
    kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).to(x.device)
    kernel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).to(x.device)
    
    kernel_x = kernel_x.view(1, 1, 3, 3).expand(x.shape[1], 1, 3, 3)
    kernel_y = kernel_y.view(1, 1, 3, 3).expand(x.shape[1], 1, 3, 3)
    
    # 计算梯度
    gx_x = F.conv2d(x, kernel_x, padding=1, groups=x.shape[1])
    gy_x = F.conv2d(x, kernel_y, padding=1, groups=x.shape[1])
    
    gx_y = F.conv2d(y, kernel_x, padding=1, groups=y.shape[1])
    gy_y = F.conv2d(y, kernel_y, padding=1, groups=y.shape[1])
    
    return F.l1_loss(gx_x, gx_y) + F.l1_loss(gy_x, gy_y)

def global_gradient_loss(a, b):
    """
    全局梯度损失
    """
    loss = 0
    for item in range(len(a)):
        feat_a = a[item]
        feat_b = b[item]
        
        if feat_a.ndim == 3:
            B, N, C = feat_a.shape
            H = int(N ** 0.5)
            if H * H == N:
                feat_a = feat_a.permute(0, 2, 1).reshape(B, C, H, H)
                feat_b = feat_b.permute(0, 2, 1).reshape(B, C, H, H)
        
        loss += gradient_loss_func(feat_a, feat_b)
        
    return loss / len(a)


def orthogonal_loss(features, device=None):
    """
    正交损失: 强制特征通道独立
    features: list of tensors [B, N, C] or [B, C, H, W]
    """
    loss = 0
    for feat in features:
        # Normalize columns
        if feat.ndim == 4:
            # [B, C, H, W] -> [B*H*W, C]
            f = feat.permute(0, 2, 3, 1).reshape(-1, feat.shape[1])
        elif feat.ndim == 3:
            # [B, N, C] -> [B*N, C]
            f = feat.reshape(-1, feat.shape[-1])
        else:
            continue
        
        # Normalize columns
        f_norm = F.normalize(f, p=2, dim=0)
        
        # Gram matrix: [C, C]
        gram = torch.mm(f_norm.t(), f_norm)
        
        # Identity matrix
        eye = torch.eye(gram.shape[0], device=gram.device)
        
        # Loss: || Gram - I ||_F
        loss += F.mse_loss(gram, eye)
        
    return loss / len(features)


# ==========================================
# 原有代码
# ==========================================

def global_l2_loss(a, b):
    """
    全局 L2 损失（MSE），保留特征幅度信息
    
    Args:
        a: encoder features (list of tensors)
        b: decoder features (list of tensors)
    
    Returns:
        loss: L2 损失
    """
    loss = 0
    for item in range(len(a)):
        a_ = a[item].detach()
        b_ = b[item]
        loss += F.mse_loss(a_, b_)
    return loss / len(a)


def global_smooth_l1_loss(a, b, beta=1.0):
    """
    全局 Smooth L1 损失 (Huber Loss)
    
    当 |a - b| < beta 时，使用 0.5 * (a - b)^2 / beta
    当 |a - b| >= beta 时，使用 |a - b| - 0.5 * beta
    
    Args:
        a: encoder features (list of tensors)
        b: decoder features (list of tensors)
        beta: L1 和 L2 的切换阈值
    
    Returns:
        loss: Smooth L1 损失
    """
    loss = 0
    for item in range(len(a)):
        a_ = a[item].detach()
        b_ = b[item]
        loss += F.smooth_l1_loss(a_, b_, beta=beta)
    return loss / len(a)



def hybrid_cosine_l2_loss(a, b, p=0.7, factor=0.1, alpha=0.5):
    """
    混合损失：Cosine + L2
    
    Args:
        a: encoder features
        b: decoder features
        p: hard mining 保留比例
        factor: 简单样本梯度缩放因子
        alpha: cosine 损失权重 (1-alpha 为 L2 权重)
    
    Returns:
        loss: alpha * cosine_loss + (1-alpha) * l2_loss
    """
    cosine_loss = global_cosine_hm_percent(a, b, p=p, factor=factor)
    l2_loss = global_l2_loss(a, b)
    return alpha * cosine_loss + (1 - alpha) * l2_loss


def staged_hybrid_loss(a, b, decoder_type='hybrid', p=0.7, factor=0.1):
    """
    阶段性混合损失：针对 Hybrid Decoder (ViT + Mamba) - Plan A
    
    ViT 前半部分 (全局)：70% Cosine + 30% L2
    Mamba 后半部分 (局部细化)：60% Cosine + 40% L2
    
    Args:
        a: encoder features (list of 8 tensors for hybrid decoder)
        b: decoder features
        decoder_type: 'hybrid', 'vit', 'mamba'
        p: hard mining 比例
        factor: 简单样本梯度缩放
    
    Returns:
        loss: 阶段性混合损失
    """
    if decoder_type != 'hybrid' or len(a) < 8:
        # 如果不是 hybrid 或者层数不够，回退到统一混合损失
        return hybrid_cosine_l2_loss(a, b, p=p, factor=factor, alpha=0.5)
    
    # 分离 ViT 层 (前半) 和 Mamba 层 (后半)
    mid = len(a) // 2
    a_vit = a[:mid]
    b_vit = b[:mid]
    a_mamba = a[mid:]
    b_mamba = b[mid:]
    
    # ViT 阶段：更偏重方向对齐 (70% Cosine, 30% L2) [Plan A]
    cos_vit = global_cosine_hm_percent(a_vit, b_vit, p=p, factor=factor)
    l2_vit = global_l2_loss(a_vit, b_vit)
    loss_vit = 0.7 * cos_vit + 0.3 * l2_vit
    
    # Mamba 阶段：平衡方向与幅度 (60% Cosine, 40% L2) [Plan A]
    cos_mamba = global_cosine_hm_percent(a_mamba, b_mamba, p=p, factor=factor)
    l2_mamba = global_l2_loss(a_mamba, b_mamba)
    loss_mamba = 0.6 * cos_mamba + 0.4 * l2_mamba
    
    # 平衡两个阶段
    return 0.5 * loss_vit + 0.5 * loss_mamba


def pure_staged_loss(a, b, decoder_type='hybrid', p=0.7, factor=0.1, 
                     vit_loss='cosine', mamba_loss='l2'):
    """
    纯阶段性损失：ViT 和 Mamba 阶段使用不同的纯损失
    
    Args:
        a: encoder features (list of tensors)
        b: decoder features
        decoder_type: 'hybrid', 'vit', 'mamba'
        p: hard mining 比例
        factor: 简单样本梯度缩放
        vit_loss: ViT 阶段使用的损失类型 ('cosine' 或 'l2')
        mamba_loss: Mamba 阶段使用的损失类型 ('cosine' 或 'l2')
    
    Returns:
        loss: 纯阶段性损失
        
    Examples:
        # ViT用Cosine, Mamba用L2
        loss = pure_staged_loss(a, b, vit_loss='cosine', mamba_loss='l2')
        
        # ViT用L2, Mamba用Cosine
        loss = pure_staged_loss(a, b, vit_loss='l2', mamba_loss='cosine')
    """
    if decoder_type != 'hybrid' or len(a) < 8:
        # 如果不是 hybrid，根据 vit_loss 参数选择损失
        if vit_loss == 'cosine':
            return global_cosine_hm_percent(a, b, p=p, factor=factor)
        else:
            return global_l2_loss(a, b)
    
    # 分离 ViT 层 (前半) 和 Mamba 层 (后半)
    mid = len(a) // 2
    a_vit = a[:mid]
    b_vit = b[:mid]
    a_mamba = a[mid:]
    b_mamba = b[mid:]
    
    # ViT 阶段：根据参数选择纯损失
    if vit_loss == 'cosine':
        loss_vit = global_cosine_hm_percent(a_vit, b_vit, p=p, factor=factor)
    elif vit_loss == 'l2':
        loss_vit = global_l2_loss(a_vit, b_vit)
    else:
        raise ValueError(f"未知的 vit_loss 类型: {vit_loss}，应为 'cosine' 或 'l2'")
    
    # Mamba 阶段：根据参数选择纯损失
    if mamba_loss == 'cosine':
        loss_mamba = global_cosine_hm_percent(a_mamba, b_mamba, p=p, factor=factor)
    elif mamba_loss == 'l2':
        loss_mamba = global_l2_loss(a_mamba, b_mamba)
    else:
        raise ValueError(f"未知的 mamba_loss 类型: {mamba_loss}，应为 'cosine' 或 'l2'")
    
    # 平衡两个阶段
    return 0.5 * loss_vit + 0.5 * loss_mamba
