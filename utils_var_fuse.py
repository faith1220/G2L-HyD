import torch
import torch.nn.functional as F


@torch.no_grad()
def compute_group_var_weights(model, loader, device, eps: float = 1e-6):
    """
    计算分组特征的方差驱动权重（1 / (var + eps)），用于自适应融合。
    返回一个 list，每个元素是 [1,1,H,W] 张量；若样本不足或异常则返回 None。
    """
    sums, sqs = None, None
    count = 0
    for imgs, _ in loader:
        imgs = imgs.to(device)
        en, _ = model(imgs)
        if sums is None:
            sums = [torch.zeros_like(e).sum(dim=0, keepdim=False) for e in en]
            sqs = [torch.zeros_like(e).sum(dim=0, keepdim=False) for e in en]
        for i, e in enumerate(en):
            # e: [B, C, H, W]
            sums[i] = sums[i] + e.sum(dim=0)
            sqs[i] = sqs[i] + (e * e).sum(dim=0)
        count += en[0].shape[0]

    if count == 0 or sums is None:
        return None

    weights = []
    for s, q in zip(sums, sqs):
        mean = s / count
        var = q / count - mean * mean
        # 在通道维求平均，得到 [H,W]，再加 eps 避免除零
        var_map = var.mean(dim=0, keepdim=False).clamp_min(0.0)
        w = 1.0 / (var_map + eps)
        weights.append(w.unsqueeze(0).unsqueeze(0))  # [1,1,H,W]
    return weights


__all__ = ["compute_group_var_weights"]
