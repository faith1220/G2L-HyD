# ============================
# models/Bayesian.py (updated)
# ============================

# === [ADD] 基础依赖（放在已有 import 后面即可） ===
import math
import numpy as np
from pathlib import Path
import torch

# === [ADD] A/B/C 所需：确保有 F / nn 可用 ===
import torch.nn.functional as F
import torch.nn as nn


# =========================
# === [A] MC-Dropout 工具 ===
# =========================
def _set_module_mode_for_mc_dropout(model: nn.Module):
    """
    仅让 Dropout 层处于 train() 以启用随机性；其余仍保持 eval() 行为。
    用法：
      model.eval(); _set_module_mode_for_mc_dropout(model)
    """
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            m.train()
        else:
            # 不改其他层（包括 LayerNorm/Conv/Mamba/Linear 等）
            pass


def mc_dropout_forward_maps(forward_fn, images, T: int):
    """
    forward_fn: 接收 images -> 返回一张基础异常热图（H×W 或 B×1×H×W）
    images: 形状 B×C×H×W
    返回：mean_map, std_map（与 forward_fn 输出同尺寸）
    """
    maps = []
    with torch.no_grad():
        for _ in range(T):
            maps.append(forward_fn(images))
    maps = torch.stack(maps, dim=0)  # T × (B×1×H×W 或 B×H×W)
    mean_map = maps.mean(dim=0)
    std_map = maps.std(dim=0)
    return mean_map, std_map


# =======================================
# === [B] 高斯原型 / 马氏距离（用在 VIB）===
# =======================================
@torch.no_grad()
def collect_vib_mu_from_batch(model: nn.Module):
    """
    从模型里抓取“最后一个 VIB 模块”的 mu（B×C×h×w 或 B×N×C）。
    前提：使用本文件的 bMlpVIBWrapper / VIBBottleneck（见下）。
    """
    vib_modules = [m for m in model.modules() if hasattr(m, '_vib_is_wrapper') and m._vib_is_wrapper]
    if not vib_modules:
        raise RuntimeError('[B] 未找到 VIB 模块，请确认已应用 VIB 补丁。')
    mu = vib_modules[-1]._last_mu  # 最近一次 forward 保存
    if mu is None:
        raise RuntimeError('[B] VIB 未进行过一次前向，无法收集 mu。')
    return mu


def _to_feat_matrix(x: torch.Tensor) -> torch.Tensor:
    """
    把特征变为 [S, C] 的二维矩阵，S 为 sample 数（像素或 token），C 为通道。
    支持形状：
      - B×C×H×W  -> (B*H*W)×C
      - B×N×C    -> (B*N)×C
    """
    if x.dim() == 4:
        B, C, H, W = x.shape
        return x.permute(0, 2, 3, 1).reshape(B * H * W, C).contiguous()
    elif x.dim() == 3:
        B, N, C = x.shape
        return x.reshape(B * N, C).contiguous()
    else:
        return x.reshape(-1, x.shape[-1]).contiguous()


def build_gaussian_prototypes_from_loader(model: nn.Module, loader, device, min_count=2000, save_path='prototypes/normal_mu_var.npz'):
    """
    用支持集/正常图像构建基于 VIB mu 的高斯原型（对角协方差）。
    """
    model.eval()
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    feats = []
    with torch.no_grad():
        for batch in loader:
            # 你已有的数据取法可能是：images, labels = batch['image'].to(device), batch['label']
            # 下面给出通用示意，按你原脚本的 batch 字段替换：
            images = batch[0].to(device) if isinstance(batch, (list, tuple)) else batch['image'].to(device)

            _ = model(images)  # 一次前向，让 VIB 模块记录 _last_mu
            mu = collect_vib_mu_from_batch(model)
            feats.append(_to_feat_matrix(mu).cpu())

    feats = torch.cat(feats, dim=0)
    # 若样本过多，随机下采样，保证统计稳定同时不爆内存
    if feats.shape[0] > min_count * 10:
        idx = torch.randperm(feats.shape[0])[:min_count * 10]
        feats = feats[idx]

    mu = feats.mean(dim=0).numpy()  # [C]
    var = feats.var(dim=0, unbiased=False).numpy()  # [C]
    np.savez(save_path, mu=mu, var=var)
    print(f'[B] 原型已保存到 {save_path}，C={mu.shape[0]}, 样本数={feats.shape[0]}')


def load_gaussian_prototypes(path: str):
    data = np.load(path)
    mu = torch.from_numpy(data['mu']).float()      # [C]
    var = torch.from_numpy(data['var']).float()    # [C]
    return mu, var


def maha_distance_map_from_mu(mu_map: torch.Tensor, mu_proto: torch.Tensor, var_proto: torch.Tensor, eps: float = 1e-6):
    """
    输入:
      mu_map: B×C×h×w 或 B×N×C 的 VIB 均值特征图
      mu_proto: [C]     var_proto: [C]
    输出:
      dist_map: B×1×h×w 或 B×N×1
    """
    if mu_map.dim() == 4:
        B, C, h, w = mu_map.shape
        diff = mu_map - mu_proto.view(1, C, 1, 1).to(mu_map.device)
        denom = (var_proto + eps).view(1, C, 1, 1).to(mu_map.device)
        dist2 = (diff * diff) / denom
        dist = dist2.sum(dim=1, keepdim=True).sqrt()
        return dist
    elif mu_map.dim() == 3:
        B, N, C = mu_map.shape
        diff = mu_map - mu_proto.view(1, 1, C).to(mu_map.device)
        denom = (var_proto + eps).view(1, 1, C).to(mu_map.device)
        dist2 = (diff * diff) / denom
        dist = dist2.sum(dim=-1, keepdim=True).sqrt()
        return dist
    else:
        raise ValueError('[B] mu_map 维度不支持')


# ==================================
# === [C] 汇总 KL（训练时加到 loss）===
# ==================================
def vib_kl_normal(mu: torch.Tensor, logvar: torch.Tensor):
    """
    KL( N(mu, sigma^2) || N(0,1) ) = 0.5 * sum( mu^2 + sigma^2 - log(sigma^2) - 1 )
    支持任意最后维为通道的张量（像素/patch 位置自动求和）。
    """
    kl = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1.0)
    # 对非通道维度求和，再对 batch 求均值，避免与图像尺寸强绑定
    while kl.dim() > 2:
        kl = kl.sum(dim=-1)
    return kl.mean()


# =============================
# === [C] VIB 包装器（新增） ===
# =============================
class bMlpVIBWrapper(nn.Module):
    """
    将现有的瓶颈 MLP/块包装起来：
      h = base(x)
      mu, logvar = Head(h)     # 形状与 h 的通道一致（自适配）
      z = mu + std * eps
      返回 z；并把 mu/logvar 暂存，供 KL 与原型统计使用
    """
    def __init__(self, base: nn.Module, beta: float = 5e-4):
        super().__init__()
        self.base = base
        self.beta = beta
        self._vib_is_wrapper = True
        self._heads_built = False
        self._last_mu = None
        self._last_logvar = None
        # 延迟初始化：根据第一次 forward 的张量维度决定用 Linear 还是 1x1 Conv
        self.mu_head = None
        self.lv_head = None

    def _build_heads(self, h: torch.Tensor):
        if h.dim() == 4:
            C = h.shape[1]
            self.mu_head = nn.Conv2d(C, C, kernel_size=1, bias=True)
            self.lv_head = nn.Conv2d(C, C, kernel_size=1, bias=True)
        elif h.dim() == 3:
            C = h.shape[-1]
            self.mu_head = nn.Linear(C, C, bias=True)
            self.lv_head = nn.Linear(C, C, bias=True)
        else:
            C = h.shape[-1]
            self.mu_head = nn.Linear(C, C, bias=True)
            self.lv_head = nn.Linear(C, C, bias=True)

        # === [FIX][DEVICE] 关键：把新建的 head 移到与输入 h 相同的设备（cuda:0 / cuda:1 / cpu）
        dev = h.device
        self.mu_head = self.mu_head.to(dev)
        self.lv_head = self.lv_head.to(dev)

        self._heads_built = True

    def forward(self, x):
        h = self.base(x)  # 保持原逻辑

        if not self._heads_built:
            self._build_heads(h)
        else:
            # === [FIX][SAFETY] 若设备仍不一致（比如某些特殊执行路径），强制对齐
            head_dev = next(self.mu_head.parameters()).device
            if head_dev != h.device:
                self.mu_head = self.mu_head.to(h.device)
                self.lv_head = self.lv_head.to(h.device)

        if h.dim() == 4:
            mu = self.mu_head(h)
            logvar = self.lv_head(h)
        elif h.dim() == 3:
            mu = self.mu_head(h)
            logvar = self.lv_head(h)
        else:
            mu = self.mu_head(h)
            logvar = self.lv_head(h)

        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps

        # 暂存，供 KL 和原型使用
        self._last_mu = mu
        self._last_logvar = logvar
        return z


def wrap_bottleneck_with_vib(model: nn.Module, beta: float):
    """
    在 model 中找到“瓶颈 MLP 类”的实例并包成 VIB。
    匹配规则：类名或属性名中包含 'bmlp' / 'mlp' / 'bottleneck'（忽略大小写）。
    如需更精确，可手动点名模块路径。
    """
    target_modules = []
    for name, m in model.named_modules():
        cls = m.__class__.__name__.lower()
        if ('bmlp' in cls or 'bottleneck' in cls or (cls == 'mlp')) and not hasattr(m, '_vib_is_wrapper'):
            target_modules.append((name, m))
    # 只包最后一个更稳（通常是解码器中间）
    if not target_modules:
        print('[C] 未自动找到瓶颈模块，跳过 VIB 包装（如需可手动指定）。')
        return model

    last_name, last_module = target_modules[-1]
    # 在父模块里替换
    parent = model
    *parents, leaf = last_name.split('.')
    for p in parents:
        parent = getattr(parent, p)
    setattr(parent, leaf, bMlpVIBWrapper(last_module, beta=beta))
    print(f'[C] 已用 VIB 包装: {last_name}')
    return model


def sum_vib_kl_from_model(model: nn.Module):
    """汇总所有 VIB wrapper 的 KL，并乘以各自 beta。"""
    total = 0.0
    for m in model.modules():
        if hasattr(m, '_vib_is_wrapper') and m._vib_is_wrapper:
            if (m._last_mu is not None) and (m._last_logvar is not None):
                total = total + m.beta * vib_kl_normal(m._last_mu, m._last_logvar)
    return total
