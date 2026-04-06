# models/memknn_head.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------- Helpers --------
def _tokens_to_grid(t: torch.Tensor) -> torch.Tensor:
    """[B,N,C] → [B,C,h,w] or pass-through if [B,C,h,w]."""
    if t.dim() == 4:
        return t
    assert t.dim() == 3, f"Expect tokens [B,N,C] or grid [B,C,h,w], got {tuple(t.shape)}"
    B, N, C = t.shape
    h = int(round(math.sqrt(N)))
    if h * h == N:
        return t.transpose(1, 2).contiguous().view(B, C, h, h)
    if (N - 1) > 0 and int(round(math.sqrt(N - 1))) ** 2 == (N - 1):
        tt = t[:, 1:, :]
        h = int(round(math.sqrt(N - 1)))
        return tt.transpose(1, 2).contiguous().view(B, C, h, h)
    raise ValueError(f"Cannot reshape tokens {tuple(t.shape)} to a square grid.")

def _resize_aa(x: torch.Tensor, size_hw):
    """抗混叠下采样 / 双线性上采样"""
    H, W = x.shape[-2:]
    h, w = size_hw
    if h < H or w < W:
        return F.interpolate(x, size=(h, w), mode='area')
    else:
        return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=False)

def _to_nchw(img):
    # 兼容多视图/tuple：取第一个视图；也可改成 flatten(0,1) 展开
    if isinstance(img, (list, tuple)):
        img = img[0]
    if isinstance(img, torch.Tensor) and img.dim() == 5:
        img = img[:, 0]  # [B,C,H,W]
    if isinstance(img, torch.Tensor) and img.dim() == 3:
        img = img.unsqueeze(0)
    assert isinstance(img, torch.Tensor) and img.dim() == 4, f"Expect NCHW, got {getattr(img,'shape',None)}"
    return img

# -------- Random projector (train-free dim reduction) --------
class RandomProjector(nn.Module):
    """
    训练免调参的随机投影：C -> D（每个层共享/独立都可，这里按“每层独立”支持动态 C）
    """
    def __init__(self, out_dim: int = 64, seed: int = 0):
        super().__init__()
        self.out_dim = out_dim
        self.seed = seed
        self.mats = {}  # {C: weight[C,D]}

    def _get_mat(self, C, device, dtype):
        key = (C, device, dtype)
        if key not in self.mats:
            g = torch.Generator(device=device)
            g.manual_seed(self.seed + C)
            W = torch.randn(C, self.out_dim, generator=g, device=device, dtype=dtype)
            W = F.normalize(W, dim=0)  # 列归一，数值更稳
            self.mats[key] = W
        return self.mats[key]

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [..., C]
        C = x.shape[-1]
        W = self._get_mat(C, x.device, x.dtype)
        return x @ W  # [..., D]

# -------- Memory builder & query --------
@torch.no_grad()
def build_memory_bank(
    model, dataset, device,
    layers=(2,6,8), from_branch='enc',  # 'enc' or 'dec'
    proj_dim=64, proj_seed=0, normalize=True,
    batch_size=8, num_workers=0, ms_scales=('1.0',), img_size_hw=None,
):
    """
    用 few-shot 正常图建立 per-layer 的 memory bank。
    返回 dict: {'bank': [L层各自的 (M,D) bank], 'proj': projector, 'layers':..., 'from':...}
    """
    projector = RandomProjector(out_dim=proj_dim, seed=proj_seed).to(device)
    bank_per_layer = [ [] for _ in layers ]

    loader = torch.utils.data.DataLoader(dataset, batch_size=min(batch_size, len(dataset)),
                                         shuffle=False, num_workers=num_workers, drop_last=False)
    for img, *_ in loader:
        img = _to_nchw(img).to(device)
        en, de = model(img)

        src_list = en if (from_branch == 'enc') else de
        assert isinstance(src_list, (list, tuple)), "model(...) should return a list/tuple per layer"

        for li, layer_idx in enumerate(layers):
            feat = src_list[layer_idx]  # [B,N,C] 或 [B,C,h,w]
            feat = _tokens_to_grid(feat)                      # [B,C,h,w]
            if img_size_hw is not None:
                feat = _resize_aa(feat, img_size_hw)          # 对齐到像素热图尺寸（可选）
            B, C, H, W = feat.shape
            f = feat.permute(0,2,3,1).reshape(-1, C)          # [B*H*W, C]
            f = projector(f)                                  # [B*H*W, D]
            if normalize:
                f = F.normalize(f, dim=-1)
            bank_per_layer[li].append(f)

    bank_per_layer = [ torch.cat(x, dim=0) if len(x)>0 else torch.empty(0, proj_dim, device=device) 
                       for x in bank_per_layer ]

    return {
        'bank': bank_per_layer,      # list of [M,D]
        'proj': projector,
        'layers': list(layers),
        'from': from_branch,
        'normalize': normalize,
        'proj_dim': proj_dim,
    }

@torch.no_grad()
def knn_heatmap(
    model, x, bank_obj, device,
    k=3, from_branch=None, layers=None, img_size_hw=None
):
    """
    计算 KNN 距离热图（cosine）：A_knn = 1 - mean(top-k sim)
    返回 [B,1,H,W]
    """
    x = _to_nchw(x).to(device)
    en, de = model(x)
    src_list = en if ((from_branch or bank_obj['from']) == 'enc') else de

    proj = bank_obj['proj']
    use_layers = layers or bank_obj['layers']
    out_maps = []

    for li, layer_idx in enumerate(use_layers):
        feat = src_list[layer_idx]             # [B,N,C] or [B,C,h,w]
        feat = _tokens_to_grid(feat)           # [B,C,h,w]
        B, C, H, W = feat.shape
        if img_size_hw is not None and (H,W) != img_size_hw:
            feat = _resize_aa(feat, img_size_hw)
            B, C, H, W = feat.shape

        q = feat.permute(0,2,3,1).reshape(B*H*W, C)   # [Q,C]
        q = proj(q)                                   # [Q,D]
        if bank_obj['normalize']:
            q = F.normalize(q, dim=-1)

        bank = bank_obj['bank'][li]                   # [M,D]
        if bank.numel() == 0:
            # 空 bank 时给 0 图（不影响融合）
            out_maps.append(torch.zeros(B,1,H,W, device=device, dtype=x.dtype))
            continue

        # 相似度：cosine（因为已归一化，直接点积）
        sim = q @ bank.t()                            # [Q, M]
        # top-k 平均
        if k > 1 and bank.shape[0] >= k:
            topk = torch.topk(sim, k=k, dim=1).values.mean(dim=1)  # [Q]
        else:
            topk = sim.max(dim=1).values
        dist = (1.0 - topk).clamp_min(0)              # [Q]
        dist = dist.view(B, H, W).unsqueeze(1)        # [B,1,H,W]
        out_maps.append(dist)

    A = torch.stack(out_maps, dim=0).mean(dim=0)      # 按层平均 [B,1,H,W]
    return A
