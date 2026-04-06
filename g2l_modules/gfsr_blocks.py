"""
Added get selfattention from all layer

Mostly copy-paster from DINO (https://github.com/facebookresearch/dino/blob/main/vision_transformer.py)
and timm library (https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py)

"""
# Copyright (c) Facebook, Inc. and its affiliates.
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import math
from functools import partial

import torch
import torch.nn as nn

from torch.nn.init import trunc_normal_

# models/bottlenecks.py
from dataclasses import dataclass
try:
    from typing import Optional, List, Tuple, Literal
except ImportError:
    from typing import Optional, List, Tuple
    from typing_extensions import Literal

import torch.nn.functional as F

def soft_stop_grad(x, gamma: float = 1.0):
    return x if gamma == 1.0 else (gamma * x + (1.0 - gamma) * x.detach())

class BatchNorm1d(nn.BatchNorm1d):
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = super(BatchNorm1d, self).forward(x)
        x = x.permute(0, 2, 1)
        return x


class ShuffleDrop(nn.Module):
    def __init__(self, p=0.):
        super(ShuffleDrop, self).__init__()
        self.p = p

    def forward(self, x):
        if self.training:
            N, P, C = x.shape
            idx = torch.randperm(N * P)
            shuffle_x = x.reshape(-1, C)[idx, :].view(x.size()).detach()
            drop_mask = torch.bernoulli(torch.ones_like(x) * self.p).bool()
            x[drop_mask] = shuffle_x[drop_mask]
        return x


class MeanDrop(nn.Module):
    def __init__(self, p=0.):
        super(MeanDrop, self).__init__()
        self.p = p

    def forward(self, x):
        if self.training:
            mean = x.mean()
            drop_mask = torch.bernoulli(torch.ones_like(x) * self.p).bool()
            x[drop_mask] = mean
        return x


class bMlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.,
                 grad=1.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.grad = grad

    def forward(self, x):
        x = self.drop(x)
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        # x = self.grad * x + (1 - self.grad) * x.detach()
        return x
    
import torch
import torch.nn as nn
import torch.nn.functional as F

class SwiGLUFFN(nn.Module):
    """
    Drop-in to replace bMlp:
    - 默认按参数量与 FFN(4×) 对齐：h = int(8/3 * in_features)
    - 提供 pre/mid/post 三处 dropout（与 bMlp 一致甚至更强）
    - 保留可选的“软停梯度”门（grad）
    """
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., grad=1.0):
        super().__init__()
        out_features   = out_features or in_features
        hidden_features = hidden_features or int(8 * in_features / 3)  # param-parity 默认
        self.up   = nn.Linear(in_features, 2 * hidden_features)
        self.down = nn.Linear(hidden_features, out_features)
        self.act  = nn.SiLU()
        self.drop_pre  = nn.Dropout(drop)
        self.drop_mid  = nn.Dropout(drop)
        self.drop_post = nn.Dropout(drop)
        self.grad = grad

    def forward(self, x):
        x = self.drop_pre(x)
        u, v = self.up(x).chunk(2, dim=-1)   # [B,*,h], [B,*,h]
        x = self.act(u) * v                   # SwiGLU
        x = self.drop_mid(x)
        x = self.down(x)
        x = self.drop_post(x)
        # 可选：软停梯度（与 bMlp 相同思路）
        # if self.grad != 1.0:
        #     x = self.grad * x + (1 - self.grad) * x.detach()
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class DropKey(nn.Module):
    """DropKey
    """

    def __init__(self, p=0.):
        super(DropKey, self).__init__()
        self.p = p

    def forward(self, attn):
        if self.training:
            m_r = torch.ones_like(attn) * self.p
            attn = attn + torch.bernoulli(m_r) * -1e12
        return attn


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        # self.attn_drop = nn.Dropout(attn_drop)
        self.attn_drop = DropKey(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        attn = self.attn_drop(attn)
        attn = attn.softmax(dim=-1)

        if attn_mask is not None:
            attn = attn.clone()
            attn[:, :, attn_mask == 0.] = 0.

        # x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = torch.matmul(attn, v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class EfficientAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = DropKey(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = torch.softmax(q, dim=-1)
        k = torch.softmax(k, dim=-2)

        context = (k.transpose(-2, -1) @ v)

        x = (q @ context).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, context


class LinearAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = nn.functional.elu(q) + 1.
        k = nn.functional.elu(k) + 1.

        attn = (q @ k.transpose(-2, -1))
        attn = self.attn_drop(attn)

        if attn_mask is not None:
            attn[:, :, attn_mask == 0.] = 0.

        attn = attn / (torch.sum(attn, dim=-1, keepdim=True))

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


class LinearAttention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        q = nn.functional.elu(q) + 1.
        k = nn.functional.elu(k) + 1.

        kv = torch.einsum('...sd,...se->...de', k, v)
        z = 1.0 / torch.einsum('...sd,...d->...s', q, k.sum(dim=-2))
        x = torch.einsum('...de,...sd,...s->...se', kv, q, z)
        x = x.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x, kv


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, attn=Attention):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn(
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False, attn_mask=None):
        if attn_mask is not None:
            y, attn = self.attn(self.norm1(x), attn_mask=attn_mask)
        else:
            y, attn = self.attn(self.norm1(x))
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if return_attention:
            return x, attn
        else:
            return x


class ConvBlock(nn.Module):
    def __init__(self, dim, kernel_size=3, mlp_ratio=4., drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.conv = SepConv(dim, kernel_size=kernel_size, act1_layer=act_layer)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, return_attention=False, attn_mask=None):
        y = self.conv(self.norm1(x))
        x = x + self.drop_path(y)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        if return_attention:
            return x, None
        else:
            return x


class FeatureJitter(nn.Module):
    def __init__(self, scale=1.):
        super().__init__()
        self.scale = scale

    def forward(self, feature_tokens):
        if self.training:
            batch_size, num_tokens, dim_channel = feature_tokens.shape
            feature_norms = feature_tokens.norm(dim=2).unsqueeze(2) / dim_channel  # B x N x 1
            jitter = torch.randn((batch_size, num_tokens, dim_channel)).to(feature_tokens.device)
            jitter = jitter * feature_norms * self.scale
            feature_tokens = feature_tokens + jitter
        return feature_tokens


class SepConv(nn.Module):
    r"""
    Inverted separable convolution from MobileNetV2: https://arxiv.org/abs/1801.04381.
    """

    def __init__(self, dim, expansion_ratio=2,
                 act1_layer=nn.GELU, act2_layer=nn.Identity,
                 bias=False, kernel_size=7,
                 **kwargs, ):
        super().__init__()
        med_channels = int(expansion_ratio * dim)
        self.pwconv1 = nn.Linear(dim, med_channels, bias=bias)
        self.act1 = act1_layer()
        self.dwconv = nn.Conv2d(
            med_channels, med_channels, kernel_size=kernel_size,
            padding=kernel_size // 2, groups=med_channels, bias=bias)  # depthwise conv
        self.act2 = act2_layer()
        self.pwconv2 = nn.Linear(med_channels, dim, bias=bias)

    def forward(self, x):
        b, hxw, c = x.shape
        h = int(math.sqrt(hxw))
        x = self.pwconv1(x)
        x = self.act1(x)
        x = x.permute(0, 2, 1).reshape(b, -1, h, h)
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1).reshape(b, hxw, -1)
        x = self.act2(x)
        x = self.pwconv2(x)
        return x

import torch
import torch.nn as nn
import torch.nn.functional as F

class _TripleDrop(nn.Module):
    def __init__(self, p=0.0, kind=None):
        super().__init__()
        # kind: 'feat' -> nn.Dropout; '2d' -> nn.Dropout2d; 其它/None -> 默认 nn.Dropout
        Drop = nn.Dropout2d if kind == "2d" else nn.Dropout
        self.pre  = Drop(p)
        self.mid  = Drop(p)
        self.post = Drop(p)

    # 兼容老写法（如果有地方用 pre_drop(...) 这类）
    def pre_drop(self, x, training=None):  return self.pre(x)
    def mid_drop(self, x, training=None):  return self.mid(x)
    def post_drop(self, x, training=None): return self.post(x)

def _soft_stop_grad(x, grad: float):
    return x if grad == 1.0 else (grad * x + (1.0 - grad) * x.detach())
class LinearGMLPTokenBlock(nn.Module):
    """
    gMLP with token Linear mixing.
    新增 token_mode：
      - 'linear'  : 原始全连接
      - 'banded'  : 只允许 |i-j|<=bandwidth 的条带（等效局部卷积）
      - 'lowrank' : 低秩分解 N->r->N，r << N
    """
    def __init__(
        self, in_features, hidden_features=None, out_features=None,
        drop=0., grad=1.0, act_layer=nn.GELU,
        num_register_tokens: int = 0, has_cls: bool = True,
        token_mode: str = "linear", bandwidth: int = 7, rank: int = 64
    ):
        super().__init__()
        out_features    = out_features or in_features
        hidden_features = hidden_features or (2 * in_features)
        assert hidden_features % 2 == 0
        mid = hidden_features // 2

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.ln  = nn.LayerNorm(mid)
        self.fc2 = nn.Linear(mid, out_features)
        self.drop_pre  = nn.Dropout(drop)
        self.drop_mid  = nn.Dropout(drop)
        self.drop_post = nn.Dropout(drop)
        self.grad = grad

        self.has_cls = has_cls
        self.num_register_tokens = num_register_tokens

        # 动态 token 混合层
        self.token_mode = token_mode
        self.bandwidth  = int(bandwidth)
        self.rank       = int(rank)
        self.token_mix  = None
        self._token_mix_N = None
        self._band_mask  = None  # [N,N] buffer-like

    def _ensure_token_mix(self, N, device, dtype):
        need_create = (self.token_mix is None) or (self._token_mix_N != N)
        if need_create:
            if self.token_mode == "lowrank":
                self.token_mix = nn.Sequential(
                    nn.Linear(N, self.rank, bias=False),  # N -> r
                    nn.Linear(self.rank, N, bias=True)    # r -> N
                )
            else:
                self.token_mix = nn.Linear(N, N, bias=True)   # linear / banded 共用
            self.token_mix.to(device=device, dtype=dtype)
            self._token_mix_N = N

        else:
            # 同步 dtype/device
            for m in (self.token_mix,) if not isinstance(self.token_mix, nn.Sequential) else self.token_mix:
                if (m.weight.device != device) or (m.weight.dtype != dtype):
                    self.token_mix.to(device=device, dtype=dtype)

        # 更新条带掩膜
        if self.token_mode == "banded":
            # [N,N]，|i-j|<=bandwidth 的位置为 1
            idx = torch.arange(N, device=device)
            band = (idx.unsqueeze(0) - idx.unsqueeze(1)).abs() <= self.bandwidth
            self._band_mask = band.to(dtype=dtype)  # 保持与权重 dtype 一致

    def _apply_token_linear(self, v):  # v: [B, mid, N]
        if self.token_mode == "linear":
            return self.token_mix(v)
        elif self.token_mode == "lowrank":
            # 顺序: N->r->N
            return self.token_mix(v)
        elif self.token_mode == "banded":
            # 用带宽掩膜约束权重（F.linear 以便使用 masked weight）
            w = self.token_mix.weight * self._band_mask  # [N,N]
            b = self.token_mix.bias
            return F.linear(v, w, b)
        else:
            raise ValueError(f"Unknown token_mode: {self.token_mode}")

    def forward(self, x, attn_mask=None, **kwargs):
        B, N_all, C = x.shape
        n_head = (1 if self.has_cls else 0) + self.num_register_tokens
        x_head  = x[:, :n_head, :] if n_head > 0 else None
        x_patch = x[:, n_head:, :] if n_head > 0 else x

        y = self.drop_pre(x_patch)
        y = self.fc1(y)
        y = self.act(y)
        u, v = y.chunk(2, dim=-1)   # [B, Np, mid]
        v = self.ln(v)
        v = v.transpose(1, 2)       # [B, mid, Np]

        self._ensure_token_mix(v.shape[-1], v.device, v.dtype)
        v = self._apply_token_linear(v)  # [B, mid, Np]

        v = v.transpose(1, 2)       # [B, Np, mid]
        y = u * v
        y = self.drop_mid(y)
        y = self.fc2(y)
        y = self.drop_post(y)
        y = _soft_stop_grad(y, self.grad)

        out = torch.cat([x_head, y], dim=1) if n_head > 0 else y
        return out
class SAB(nn.Module):
    """
    gMLP with depthwise token mixing.
    - 优先使用 2D depthwise conv（保持2D邻接），无法整平方时回退 1D。
    - 仍保留三处 dropout 与 soft-stop-grad。
    """
    def __init__(
        self, in_features, hidden_features=None, out_features=None,
        drop=0., grad=1.0, act_layer=nn.GELU, dw_kernel: int = 5,
        num_register_tokens: int = 0, has_cls: bool = True
    ):
        super().__init__()
        out_features    = out_features or in_features
        hidden_features = hidden_features or (2 * in_features)
        assert hidden_features % 2 == 0
        mid = hidden_features // 2

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.ln  = nn.LayerNorm(mid)

        # 两条路径都建好，前向时按形状选择
        self.dw1d = nn.Conv1d(mid, mid, kernel_size=dw_kernel,
                              padding=dw_kernel // 2, groups=mid, bias=True)
        self.dw2d = nn.Conv2d(mid, mid, kernel_size=dw_kernel,
                              padding=dw_kernel // 2, groups=mid, bias=True)

        self.fc2 = nn.Linear(mid, out_features)
        self.drop_pre  = nn.Dropout(drop)
        self.drop_mid  = nn.Dropout(drop)
        self.drop_post = nn.Dropout(drop)
        self.grad = grad

        self.has_cls = has_cls
        self.num_register_tokens = num_register_tokens

    def forward(self, x, attn_mask=None, **kwargs):
        B, N_all, C = x.shape
        n_head = (1 if self.has_cls else 0) + self.num_register_tokens
        x_head  = x[:, :n_head, :] if n_head > 0 else None
        x_patch = x[:, n_head:, :] if n_head > 0 else x

        y = self.drop_pre(x_patch)
        y = self.fc1(y)
        y = self.act(y)
        u, v = y.chunk(2, dim=-1)          # [B, Np, mid]
        v = self.ln(v)                      # 归一化有助于稳定

        # --- 2D token mixing（优先） ---
        Np = v.shape[1]
        side = int(Np ** 0.5)
        if side * side == Np:
            # [B, Np, mid] -> [B, mid, H, W]
            v2 = v.view(B, Np, -1).transpose(1, 2).contiguous().view(B, -1, side, side)
            v2 = self.dw2d(v2)
            v  = v2.flatten(2).transpose(1, 2).contiguous()   # 回到 [B, Np, mid]
        else:
            # --- 回退：1D 沿 token 序列 ---
            v1 = v.transpose(1, 2)           # [B, mid, Np]
            v1 = self.dw1d(v1)
            v  = v1.transpose(1, 2)

        y = u * v
        y = self.drop_mid(y)
        y = self.fc2(y)
        y = self.drop_post(y)
        y = _soft_stop_grad(y, self.grad)

        out = torch.cat([x_head, y], dim=1) if n_head > 0 else y
        return out

# 兼容旧接口：DWConvGMLPTokenBlock 等价于 SAB
class DWConvGMLPTokenBlock(SAB):
    def __init__(
        self, in_features, hidden_features=None, out_features=None,
        drop=0., grad=1.0, act_layer=nn.GELU, dw_kernel: int = 5,
        num_register_tokens: int = 0, has_cls: bool = True
    ):
        super().__init__(
            in_features=in_features,
            hidden_features=hidden_features,
            out_features=out_features,
            drop=drop,
            grad=grad,
            act_layer=act_layer,
            dw_kernel=dw_kernel,
            num_register_tokens=num_register_tokens,
            has_cls=has_cls
        )

# ---------------------------
# 4) gMLP（带 Spatial Gating Unit）
# 参考 gMLP: x -> fc1 -> act -> split(u,v) -> v 做 token 维线性 -> u * v -> fc2
# 输入形状: [B, N, C]
# ---------------------------
class gMLPBlock(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 drop=0.0, grad=1.0, act_layer=nn.GELU):
        super().__init__()
        out_features    = out_features or in_features
        hidden_features = hidden_features or (in_features * 2)
        assert hidden_features % 2 == 0

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()

        self.sgu_ln = nn.LayerNorm(hidden_features // 2)

        # 动态构建的 token 线性层（对 token 维 N 做线性）
        self.token_mix = None
        self._token_mix_N = None  # 记录当前 N

        self.fc2 = nn.Linear(hidden_features // 2, out_features)
        self.do  = _TripleDrop(drop, kind="feat")
        self.grad = grad

    def _ensure_token_mix(self, N, device, dtype):
        """确保 token_mix 存在，且在正确的 device/dtype 上。"""
        need_create = (self.token_mix is None) or (self._token_mix_N != N)
        if need_create:
            layer = nn.Linear(N, N, bias=True)
            layer.to(device=device, dtype=dtype)
            self.token_mix = layer
            self._token_mix_N = N
        else:
            # 已存在但可能被 model.to(...) 搬过；若不匹配则同步
            if (self.token_mix.weight.device != device) or (self.token_mix.weight.dtype != dtype):
                self.token_mix.to(device=device, dtype=dtype)

    def forward(self, x):
        # x: [B, N, C]
        B, N, C = x.shape

        x = self.do.pre_drop(x, self.training)
        x = self.fc1(x)
        x = self.act(x)

        u, v = x.chunk(2, dim=-1)        # [B, N, H/2], [B, N, H/2]

        # SGU: 在 token 维做线性
        v = self.sgu_ln(v)
        v = v.transpose(1, 2)            # [B, H/2, N]  —— 线性作用在最后一维 N 上

        # 关键修复：创建/搬运到与 v 一致的 device/dtype
        self._ensure_token_mix(N=v.shape[-1], device=v.device, dtype=v.dtype)

        v = self.token_mix(v)            # [B, H/2, N]
        v = v.transpose(1, 2)            # [B, N, H/2]

        x = u * v
        x = self.do.mid_drop(x, self.training)
        x = self.fc2(x)
        x = self.do.post_drop(x, self.training)
        x = _soft_stop_grad(x, self.grad)
        return x
    # Counterfactual GEGLU Bottleneck GEGLUFFN
class CGB(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None,
                 drop=0.0, grad=1.0):
        super().__init__()
        out_features   = out_features or in_features
        hidden_features = hidden_features or int(8 * in_features / 3)
        self.up   = nn.Linear(in_features, 2 * hidden_features)
        self.down = nn.Linear(hidden_features, out_features)
        self.act  = nn.GELU()
        self.do   = _TripleDrop(drop, kind="feat")
        self.grad = grad

    def forward(self, x):
        # x: [B, N, C]
        x = self.do.pre_drop(x, self.training)
        u, v = self.up(x).chunk(2, dim=-1)
        x = self.act(u) * v
        x = self.do.mid_drop(x, self.training)
        x = self.down(x)
        x = self.do.post_drop(x, self.training)
        x = _soft_stop_grad(x, self.grad)
        return x


# --------------------------- 通用小工具 ---------------------------

def infer_hw_from_n(n: int) -> Tuple[int, int]:
    """从 token 数量 n 推断 (H, W)；优先取正方形网格，否则退回最接近正方的 (h, w)。"""
    root = int(math.sqrt(n))
    if root * root == n:
        return root, root
    # 找到 h*w == n 的最接近正方形的因子
    best = (1, n)
    best_diff = n - 1
    for h in range(1, root + 1):
        if n % h == 0:
            w = n // h
            if abs(w - h) < best_diff:
                best_diff = abs(w - h)
                best = (h, w)
    return best


def tokens_to_map(x: torch.Tensor) -> torch.Tensor:
    """[B, N, C] -> [B, C, H, W]"""
    B, N, C = x.shape
    H, W = infer_hw_from_n(N)
    return x.transpose(1, 2).reshape(B, C, H, W)


def map_to_tokens(x: torch.Tensor) -> torch.Tensor:
    """[B, C, H, W] -> [B, N, C]"""
    B, C, H, W = x.shape
    return x.flatten(2).transpose(1, 2)


# --------------------------- Dropout 预热 ---------------------------

@dataclass
class DropoutSchedule:
    start_p: float = 0.0
    end_p: float = 0.2
    warmup_steps: int = 1000

    def value_at(self, step: int) -> float:
        if self.warmup_steps <= 0:
            return float(self.end_p)
        s = max(0, min(step, self.warmup_steps))
        t = float(s) / float(self.warmup_steps)
        return (1.0 - t) * float(self.start_p) + t * float(self.end_p)


# --------------------------- 1) Noisy-GEGLU bMLP ---------------------------

class NoisyGEGLUBottleneck(nn.Module):
    """
    bMLP 变体：C -> 2H (GEGLU) -> H -> C
    - 三处 dropout（输入前/中/输出后），p 支持线性预热
    - 可选输出部分断梯度稳定训练
    - 缺省维度贴合 ViT-B/14: C=768, H≈(8/3)C=2048
    """
    def __init__(
        self,
        in_features: int = 768,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        dropout_schedule: Optional[DropoutSchedule] = None,
        detach_ratio: float = 0.0,
    ) -> None:
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or int(8 * in_features / 3)  # ≈ 2048 for C=768

        self.up = nn.Linear(in_features, 2 * hidden_features)  # -> (u, v)
        self.down = nn.Linear(hidden_features, out_features)
        self.act = nn.GELU()

        self.dropout_schedule = dropout_schedule or DropoutSchedule()
        self.register_buffer("_current_p", torch.tensor(self.dropout_schedule.start_p), persistent=False)
        self.drop_in = nn.Dropout(self.current_p)
        self.drop_mid = nn.Dropout(self.current_p)
        self.drop_out = nn.Dropout(self.current_p)

        self.detach_ratio = float(detach_ratio)
        self._step = 0

    @property
    def current_p(self) -> float:
        return float(self._current_p.item())

    def set_dropout(self, p: float) -> None:
        p = float(max(0.0, min(1.0, p)))
        if abs(p - self.current_p) < 1e-8:
            return
        self._current_p.fill_(p)
        self.drop_in.p = p
        self.drop_mid.p = p
        self.drop_out.p = p

    def step_scheduler(self, steps: int = 1) -> None:
        self._step += int(max(0, steps))
        self.set_dropout(self.dropout_schedule.value_at(self._step))

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, N, C]
        x = self.drop_in(x)
        u, v = self.up(x).chunk(2, dim=-1)
        h = self.act(u) * v
        h = self.drop_mid(h)
        y = self.down(h)
        y = self.drop_out(y)
        if self.detach_ratio > 0:
            y = (1.0 - self.detach_ratio) * y + self.detach_ratio * y.detach()
        return y


# --------------------------- 2) Feature Jitter ---------------------------

class FeatureJitter(nn.Module):
    """
    加性高斯噪声，训练态生效：x <- x + N(0, sigma^2)
    建议：MVTec/VisA sigma=0.1；Real-IAD sigma=0.1~0.15
    """
    def __init__(self, sigma: float = 0.1, p: float = 1.0):
        super().__init__()
        self.sigma = float(sigma)
        self.p = float(p)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training and torch.rand(1, device=x.device) < self.p and self.sigma > 0:
            return x + torch.randn_like(x) * self.sigma
        return x


# --------------------------- 3) Mask Token ---------------------------

class MaskTokenBottleneck(nn.Module):
    """
    随机以 mask_ratio 替换一部分 token 为可学习的 [MASK] 向量。
    位置无需还原到原顺序；与 Loose Reconstruction 兼容。
    """
    def __init__(self, dim: int, mask_ratio: float = 0.3, learnable: bool = True):
        super().__init__()
        self.mask_ratio = float(mask_ratio)
        self.token = nn.Parameter(torch.zeros(1, 1, dim)) if learnable else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # [B, N, C]
        if not self.training or self.mask_ratio <= 0:
            return x
        B, N, C = x.shape
        keep = int(round((1.0 - self.mask_ratio) * N))
        idx = torch.rand(B, N, device=x.device).argsort(dim=1)
        keep_idx = idx[:, :keep]
        x_keep = torch.gather(x, 1, keep_idx[..., None].expand(-1, -1, C))
        if self.token is not None:
            x_mask = self.token.expand(B, N - keep, C)
        else:
            x_mask = torch.zeros(B, N - keep, C, device=x.device, dtype=x.dtype)
        return torch.cat([x_keep, x_mask], dim=1)


# --------------------------- 4) H-FPN 多层融合瓶颈 ---------------------------

class HFPNBottleneck(nn.Module):
    """
    将 8 个中间层的 token 特征融合到统一网格 (h, w)，再拼回 [B, N, C]。
    - in_dim: 输入各层通道（ViT-B/14: 768）
    - out_dim: 融合后通道，建议与 in_dim 持平（768）或略降（384）
    - out_hw: 统一到 (28,28) 对应 392 crop；不匹配时自动插值
    """
    def __init__(self, in_dim: int = 768, out_dim: int = 768, out_hw: Tuple[int, int] = (28, 28), num_levels: int = 8):
        super().__init__()
        self.num_levels = num_levels
        self.proj = nn.ModuleList([nn.Conv2d(in_dim, out_dim, 1) for _ in range(num_levels)])
        self.dw = nn.Conv2d(out_dim * num_levels, out_dim * num_levels, 3, padding=1, groups=out_dim * num_levels)
        self.fuse = nn.Conv2d(out_dim * num_levels, out_dim, 1)
        self.out_hw = out_hw

    def forward(self, feats_tok: List[torch.Tensor]) -> torch.Tensor:
        """
        feats_tok: 长度 = num_levels 的列表，每项 [B, N, C]，通常取 ViT 中间 8 层
        return: [B, N_out, out_dim]，N_out = out_hw[0]*out_hw[1]
        """
        assert len(feats_tok) == self.num_levels, f"Expect {self.num_levels} levels, got {len(feats_tok)}"
        xs = []
        for i, x in enumerate(feats_tok):
            B, N, C = x.shape
            m = tokens_to_map(x)                  # [B, C, H, W]
            m = self.proj[i](m)                   # [B, out_dim, H, W]
            m = F.interpolate(m, size=self.out_hw, mode="bilinear", align_corners=False)
            xs.append(m)
        x = torch.cat(xs, dim=1)                  # [B, out_dim*num_levels, h, w]
        x = self.dw(x)
        x = self.fuse(x)                          # [B, out_dim, h, w]
        return map_to_tokens(x)                   # [B, h*w, out_dim]


# --------------------------- 5) HVQ（VQ-EMA）瓶颈 ---------------------------

class HVQBottle(nn.Module):
    """
    向量量化瓶颈（EMA 码本），返回量化特征与 VQ 损失：
      z_q_st, loss_vq = hvq(x)
    - K: 码本大小（MVTec/VisA 256~512；Real-IAD 512~1024）
    - beta: commitment 权重（0.25 常用）
    - ema_decay: 码本更新的 EMA 衰减
    """
    def __init__(self, dim: int = 768, K: int = 512, beta: float = 0.25, ema_decay: float = 0.99, eps: float = 1e-5):
        super().__init__()
        self.dim = dim
        self.K = K
        self.beta = beta
        self.ema_decay = ema_decay
        self.eps = eps

        self.embed = nn.Embedding(K, dim)
        nn.init.normal_(self.embed.weight, mean=0.0, std=1.0 / dim)

        self.register_buffer("ema_cluster_size", torch.zeros(K))
        self.register_buffer("ema_embed", torch.zeros(K, dim))

    @torch.no_grad()
    def _ema_update(self, x_flat: torch.Tensor, indices: torch.Tensor):
        # x_flat: [BN, C], indices: [BN]
        # 统计当前 batch 的码字使用频次与向量和
        one_hot = F.one_hot(indices, num_classes=self.K).type_as(x_flat)  # [BN, K]
        batch_cluster = one_hot.sum(dim=0)                                # [K]
        batch_embed = one_hot.t() @ x_flat                                # [K, C]

        self.ema_cluster_size.mul_(self.ema_decay).add_(batch_cluster, alpha=1 - self.ema_decay)
        self.ema_embed.mul_(self.ema_decay).add_(batch_embed, alpha=1 - self.ema_decay)

        # 归一化，避免空簇
        n = self.ema_cluster_size.sum()
        cluster_size = ((self.ema_cluster_size + self.eps) / (n + self.K * self.eps)) * n
        embed_normalized = self.ema_embed / cluster_size.unsqueeze(1).clamp_min(self.eps)

        self.embed.weight.copy_(embed_normalized)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: [B, N, C]  ->  z_q_st: [B, N, C], loss_vq: scalar
        """
        B, N, C = x.shape
        x_flat = x.reshape(-1, C)                      # [BN, C]
        # 计算 L2 距离并取最近码字
        # ||x - e||^2 = ||x||^2 + ||e||^2 - 2 x·e
        x2 = (x_flat ** 2).sum(dim=1, keepdim=True)    # [BN,1]
        e = self.embed.weight                          # [K, C]
        e2 = (e ** 2).sum(dim=1)                       # [K]
        dist = x2 + e2.unsqueeze(0) - 2 * x_flat @ e.t()
        indices = dist.argmin(dim=1)                   # [BN]
        z_q = e[indices].view(B, N, C)

        # Straight-through estimator
        z_q_st = x + (z_q - x).detach()

        # VQ 损失：commitment + codebook（等价使用 EMA 后只保留 commitment）
        loss_commit = self.beta * F.mse_loss(x.detach(), z_q)
        loss_codebk = F.mse_loss(z_q, x.detach())
        loss_vq = loss_commit + loss_codebk

        if self.training:
            with torch.no_grad():
                self._ema_update(x_flat, indices)

        return z_q_st, loss_vq


# --------------------------- 简单工厂 ---------------------------

def build_bottleneck(
    variant: Literal["noisy_geglu", "jitter", "mask", "hfpn", "hvq"],
    **kwargs
) -> nn.Module:
    if variant == "noisy_geglu":
        return NoisyGEGLUBottleneck(**kwargs)
    if variant == "jitter":
        return FeatureJitter(**kwargs)
    if variant == "mask":
        return MaskTokenBottleneck(**kwargs)
    if variant == "hfpn":
        return HFPNBottleneck(**kwargs)
    if variant == "hvq":
        return HVQBottle(**kwargs)
    raise ValueError(f"Unknown bottleneck variant: {variant}")

# ====== 新增：RSC Reconstruction Head ======
class RSCReconstructionHead(nn.Module):
    """
    RSC 论文中的 Reconstruction Head (ICLR 2024)
    - 轻量特征转换模块，专为少样本设计
    - 输入/输出维度可配置，自动处理特征图/向量输入
    """
    def __init__(self, input_dim, output_dim=None, dropout=0.2):
        """
        Args:
            input_dim: 输入特征维度 (e.g., 768)
            output_dim: 输出特征维度 (默认与 input_dim 相同)
            dropout: Dropout 概率 (少样本下建议 0.1-0.3)
        """
        super().__init__()
        output_dim = output_dim or input_dim  # 默认输出维度与输入相同
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),  # 隐藏层维度减半
            nn.BatchNorm1d(input_dim // 2),        # 关键：防止少样本过拟合
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim // 2, output_dim)
        )
    
    def forward(self, x):
        """兼容特征图 [B,C,H,W] 和特征向量 [B,C]"""
        if x.dim() > 2:  # 特征图输入
            x = x.view(x.size(0), -1)
        return self.mlp(x)
