# Dinomaly/models_mambaAD/hybrid_losses.py
import torch
import torch.nn.functional as F

def multiscale_mse(feats, dec_feats):
    loss = 0.
    for f, d in zip(feats, dec_feats):
        f_ = F.interpolate(f, size=d.shape[-2:], mode='bilinear', align_corners=False)
        loss = loss + F.mse_loss(d, f_)
    return loss

def contrastive_memory_loss(query_g, mem_g, temp=0.07):
    """
    query_g: [B,D] ; mem_g: [K,D]
    用 mem_g 中相似度最大的视作正样，其余为负样本（简单近似）
    """
    if mem_g is None or mem_g.numel() == 0:
        return query_g.new_zeros(())
    sim = F.cosine_similarity(query_g.unsqueeze(1), mem_g.unsqueeze(0), dim=-1)  # [B,K]
    pos = sim.max(dim=1, keepdim=True)[0]                                        # [B,1]
    logits = torch.cat([pos, sim], dim=1) / temp
    labels = torch.zeros(query_g.size(0), dtype=torch.long, device=query_g.device)
    return F.cross_entropy(logits, labels)

def hybrid_loss(feats, dec_feats, query_g, mem_g, alpha=0.7, beta=0.3):
    l_rec = multiscale_mse(feats, dec_feats)
    l_con = contrastive_memory_loss(query_g, mem_g)
    return alpha * l_con + beta * l_rec, {'rec': float(l_rec.detach()), 'con': float(l_con.detach())}
