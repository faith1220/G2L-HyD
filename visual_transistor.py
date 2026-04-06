import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import warnings
warnings.filterwarnings("ignore")

# ---- 标准库 ----
import math
import random
import logging
import argparse
import copy
import itertools
import functools

# ---- 第三方 ----
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from PIL import Image as _PILImage
from ptflops import get_model_complexity_info
from sklearn.metrics import roc_auc_score, average_precision_score
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets import ImageFolder

# ---- 项目内模块 ----
from dataset import get_data_transforms, get_strong_transforms, MVTecDataset
from g2l_modules.g2l_model import G2LHyD, G2LHyDv2, DALRBlock
from g2l_modules import encoder
from dinov1.utils import trunc_normal_
from g2l_modules.gfsr_blocks import (
    GFSRBlock, bMlp, Attention, LinearAttention, LinearAttention,
    ConvBlock, FeatureJitter, CGB
)
from optimizers import StableAdamW
# ---- 项目内工具函数（统一从 utils 导入）----
from utils import (
    # 训练/评测
    evaluation_batch, global_cosine, regional_cosine_hm_percent,
    global_cosine_hm_percent, WarmCosineScheduler, compute_image_score_from_heatmap,
    cal_anomaly_maps
)

# [ADDED] Missing function implementation locally
def build_g2r_maps(en_groups, de_groups, num_reg_tokens=4, fuse="mean"):
    """
    Construct Global-to-Local residual maps.
    en_groups: [E_low, E_high], each is [B, T, C]
    de_groups: [D_low, D_high], each is [B, T, C]
    """
    E_low, E_high = en_groups
    D_low, D_high = de_groups
    
    # Cosine distance
    def _cosine_map(a, b):
        # a, b: [B, T, C]
        # output: [B, 1, H, W]
        # Remove register tokens if present
        if num_reg_tokens > 0:
            a = a[:, num_reg_tokens:, :]
            b = b[:, num_reg_tokens:, :]
        
        B, N, C = a.shape
        H = W = int(math.sqrt(N))
        
        sim = F.cosine_similarity(a, b, dim=-1)  # [B, N]
        dist = 1 - sim
        return dist.view(B, 1, H, W)

    H_low = _cosine_map(E_low, D_low)
    H_high = _cosine_map(E_high, D_high)
    
    # Resize high to low size if needed, or vice-versa? Usually we upscale to largest common size or image size
    # Here we assume they are same grid size for simplicity, or we upscale low to high
    if H_low.shape[-1] != H_high.shape[-1]:
        H_low = F.interpolate(H_low, size=H_high.shape[-2:], mode='bilinear', align_corners=False)
        
    if fuse == "mean":
        H_fused = (H_low + H_high) / 2
    else:
        H_fused = torch.max(H_low, H_high)
        
    return H_low, H_high, H_fused


_VIZ_FROM = "utils"  # 仅用于日志提示

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

class ResidualScale(nn.Module):
    """
    y = x + gamma * (f(x) - x)
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
    子模块输出后做一个 LayerNorm（eps更小）
    """
    def __init__(self, module: nn.Module, dim: int, eps: float = 1e-6):
        super().__init__()
        self.module = module
        self.ln = nn.LayerNorm(dim, eps=eps)

    def forward(self, x, **kwargs):
        y = self.module(x, **kwargs)
        return self.ln(y)

# ---------- 小工具：把 [B,C,H,W] or [B,T,C] 统一成 token [B,T,C] ----------
def _to_tokens(x: torch.Tensor):
    if x.dim() == 3:
        return x                        # [B,T,C]
    elif x.dim() == 4:
        B, C, H, W = x.shape
        return x.flatten(2).transpose(1, 2)  # [B,HW,C]
    else:
        raise ValueError(f"Unsupported feature shape {x.shape}")

def _group_mean_tokens(en_list, de_list, group_idx):
    """
    en_list/de_list: List[Tensor]  每个元素 [B,T,C] or [B,C,H,W]
    group_idx: e.g., [0,1,2,3]
    return: E_group, D_group as [B,T,C] (组内均值后对齐最小T)
    """
    assert len(en_list) >= max(group_idx)+1 and len(de_list) >= max(group_idx)+1
    e_sel = [_to_tokens(en_list[i]) for i in group_idx]
    d_sel = [_to_tokens(de_list[i]) for i in group_idx]
    # 对齐 token 数（不同层token数可能略有差异，取最小）
    minT = min([t.shape[1] for t in e_sel + d_sel])
    e_sel = [t[:, :minT, :] for t in e_sel]
    d_sel = [t[:, :minT, :] for t in d_sel]
    E = torch.stack(e_sel, dim=0).mean(0)  # [B,T,C]
    D = torch.stack(d_sel, dim=0).mean(0)
    return E, D

# ------------------------- 训练主函数 -------------------------
def train(item_list):
    setup_seed(1)

    total_iters = 30000
    image_size = 448
    crop_size  = 392

    data_transform, gt_transform = get_data_transforms(image_size, crop_size)

    train_data_list, test_data_list = [], []

    # ---------- Few-shot 设置 ----------
    shots_per_class = args.shots if args.shots in (1, 2, 4) else 1

    for i, item in enumerate(item_list):
        train_path = os.path.join(args.data_path, item, 'train')
        test_path  = os.path.join(args.data_path, item)

        train_data = ImageFolder(root=train_path, transform=data_transform)

        # --- few-shot 子采样 ---
        sorted_samples = train_data.samples
        rng = random.Random(1 + i)
        if len(sorted_samples) >= shots_per_class:
            keep_samples = rng.sample(sorted_samples, k=shots_per_class)
        else:
            keep_samples = sorted_samples
        train_data.samples = keep_samples[:]
        train_data.imgs    = keep_samples[:]

        # 重写类别索引为当前类 i
        train_data.classes = [item]
        train_data.class_to_idx = {item: i}
        train_data.samples = [(p, i) for (p, _) in train_data.samples]
        train_data.imgs    = train_data.samples[:]
        train_data.targets = [i] * len(train_data.samples)

        # 测试集
        test_data = MVTecDataset(root=test_path, transform=data_transform,
                                 gt_transform=gt_transform, phase="test")

        train_data_list.append(train_data)
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

    # ★ register tokens 固定为 4（可视化/热力图会裁掉它们）
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
    def make_vit_block(dim, heads):
        return GFSRBlock(
            dim=dim,
            num_heads=heads,
            mlp_ratio=4.,
            qkv_bias=True,
            norm_layer=functools.partial(nn.LayerNorm, eps=1e-8),
            attn=LinearAttention
        )

    def make_decoder(kind, dim, heads, pattern=None, depth=8, num_reg_tokens=4):
        blocks = []
        if kind == 'vit':
            for _ in range(depth):
                blocks.append(make_vit_block(dim, heads))
        elif kind == 'mamba':
            for _ in range(depth):
                # [FIX] num_scan_dirs=2 to match checkpoint
                blk = DALRBlock(embed_dim=dim, num_register_tokens=num_reg_tokens, num_scan_dirs=2)
                blk = ResidualScale(blk, init_scale=0.2)
                blk = PostNorm(blk, dim)
                blocks.append(blk)
        else:  # hybrid
            pat = [s.strip().lower() for s in (pattern or 'v,m,v,m,v,m,v,m').split(',')]
            assert len(pat) == depth
            for p in pat:
                if p == 'm':
                    # [FIX] num_scan_dirs=2 to match checkpoint
                    blk = DALRBlock(embed_dim=dim, num_register_tokens=num_reg_tokens, num_scan_dirs=2)
                    blk = ResidualScale(blk, init_scale=0.2)
                    blk = PostNorm(blk, dim)
                elif p == 'v':
                    blk = make_vit_block(dim, heads)
                else:
                    raise ValueError(f"未知标记: {p}")
                blocks.append(blk)
        return nn.ModuleList(blocks)

    # ========== Bottleneck & Decoder ==========
    bottleneck = nn.ModuleList([
        CGB(in_features=embed_dim,
            hidden_features=int(2.67 * embed_dim),
            out_features=embed_dim,
            drop=0.2,
            grad=0.7)
    ])

    decoder = make_decoder(args.decoder, embed_dim, num_heads,
                           pattern=args.hybrid_pat, depth=args.decode_depth,
                           num_reg_tokens=num_reg_tokens)
    print_fn(f"[decoder] using: {args.decoder} | "
             f"{'pattern=' + args.hybrid_pat if args.decoder=='hybrid' else 'pure'} | "
             f"depth={args.decode_depth} | num_reg_tokens={num_reg_tokens}")

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
        lr = lr * 0.5
        warmup = max(warmup, 500)
    optimizer = StableAdamW([{'params': trainable.parameters()}],
                            lr=lr, betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=True, eps=1e-10)
    lr_scheduler = WarmCosineScheduler(
        optimizer, base_value=lr, final_value=args.lr_end, total_iters=total_iters, warmup_iters=warmup
    )

    print_fn('train image number: {}'.format(len(train_data)))
    print_fn(f'few-shot setting: {shots_per_class} per class')
    print_fn(f'viz from: {_VIZ_FROM}')

    # [ADDED] Checkpoint Loading
    if args.load_ckpt:
        ckpt_path = os.path.expanduser(args.load_ckpt)
        print_fn(f"[ckpt] Loading from {ckpt_path} ...")
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        # Handle dict or direct model
        state_dict = state['model'] if 'model' in state else state
        # Remove module. prefix if present (DDP)
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k[7:] if k.startswith('module.') else k
            new_state_dict[name] = v
        missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
        print_fn(f"[ckpt] Loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    # [ADDED] Failure Case Analysis Mode
    if getattr(args, 'failure_case_mode', False):
         # --- Failure Case Analysis Mode ---
        print_fn("[Mode] Failure Case Analysis")
        model.eval()
        
        # Data structures to store results
        class_metrics = {}  # {class_name: auroc}
        failure_cases = []  # List of (score_error, image_path, gt_path, pred_score, gt_label)
        
        # Helper to save WinCLIP-style visualization
        def plot_failure_case_winclip_style(img_t, gt_t, pred_map, save_path):
            """
            img_t: [C, H, W] tensor (normalized)
            gt_t: [1, H, W] tensor (binary)
            pred_map: [1, H, W] tensor (0-1 score)
            """
            # Denormalize image (assuming ImageNet stats)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(img_t.device)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(img_t.device)
            img = img_t * std + mean
            img = img.clamp(0, 1).permute(1, 2, 0).cpu().numpy()
            img = (img * 255).astype(np.uint8)
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            # Process GT
            gt = gt_t.squeeze().cpu().numpy()
            gt_vis = np.zeros_like(img)
            gt_vis[gt > 0.5] = [255, 255, 255] # White for defect
            
            # Process Pred Map
            pred = pred_map.squeeze().cpu().numpy()
            pred_norm = (pred - pred.min()) / (pred.max() - pred.min() + 1e-8)
            heatmap = cv2.applyColorMap(np.uint8(pred_norm * 255), cv2.COLORMAP_JET)
            
            # Overlay
            overlay = cv2.addWeighted(img_bgr, 0.6, heatmap, 0.4, 0)
            
            # Combine horizontally: Original | GT | Heatmap | Overlay
            # Resize all to same height if needed (they are same here)
            combined = np.hstack([img_bgr, gt_vis, heatmap, overlay])
            
            cv2.imwrite(save_path, combined)

        all_results = {} # Store per class
        
        with torch.no_grad():
            for item, test_data in zip(item_list, test_data_list):
                print_fn(f"Analyzing class: {item}...")
                test_dataloader = torch.utils.data.DataLoader(
                    test_data, batch_size=1, shuffle=False, num_workers=args.workers
                )
                
                gt_list_px = []
                pr_list_px = []
                
                item_failures = []
                
                for idx, (img, gt, label, img_path) in enumerate(test_dataloader):
                    img = img.to(device)
                    gt = gt.to(device)
                    
                    en, de = model(img)
                    anomaly_map, _ = cal_anomaly_maps(
                         en, de, 
                         img.shape[-1],
                         weights=None,
                         fusion_mode="mean"
                    )
                    
                    # Metrics Calculation
                    gt_np = gt.cpu().numpy().astype(int).ravel()
                    pr_np = anomaly_map.cpu().numpy().ravel()
                    
                    gt_list_px.extend(gt_np)
                    pr_list_px.extend(pr_np)
                    
                    # Identify Failure Cases
                    # For simplicity, we define "failure" as high score on normal sample (False Positive)
                    # or low score on anomalous sample (False Negative).
                    # Since we don't have a threshold yet, we collect raw data first.
                    
                    item_failures.append({
                        'img': img.cpu(),
                        'gt': gt.cpu(),
                        'pred': anomaly_map.cpu(),
                        'label': label.item(),
                        'path': img_path[0],
                        'img_score': anomaly_map.max().item()
                    })
                    
                # Calculate Class Metric
                auroc_px = roc_auc_score(gt_list_px, pr_list_px)
                class_metrics[item] = auroc_px
                print_fn(f"Class {item}: P-AUROC = {auroc_px:.4f}")
                
                # Store data for later sorting
                all_results[item] = item_failures

        # 1. Rank Classes
        sorted_classes = sorted(class_metrics.items(), key=lambda x: x[1])
        worst_class = sorted_classes[0][0]
        print_fn(f"\nWorst Performing Class: {worst_class} (AUROC: {sorted_classes[0][1]:.4f})")
        
        # 2. Select Failure Cases
        # We focus on the worst class, finding False Negatives (Logical Anomalies often appear here)
        # Find anomalous samples with LOWEST max scores (Hardest to detect)
        
        target_failures = all_results[worst_class]
        anomalies = [x for x in target_failures if x['label'] == 1]
        # Sort by image score ascending (Lowest score = most looked like normal = False Negative)
        anomalies.sort(key=lambda x: x['img_score'])
        
        top_failures = anomalies[:args.viz_n]
        
        save_root = os.path.join(args.save_dir, "failure_cases", worst_class)
        os.makedirs(save_root, exist_ok=True)
        
        print_fn(f"Saving top {len(top_failures)} failure cases to {save_root}...")
        
        for i, fail in enumerate(top_failures):
            fname = os.path.basename(fail['path'])
            save_path = os.path.join(save_root, f"rank{i}_{fname}")
            plot_failure_case_winclip_style(
                fail['img'][0], 
                fail['gt'][0], 
                fail['pred'][0], 
                save_path
            )
        
        # Also save a summary text file
        with open(os.path.join(args.save_dir, "failure_summary.txt"), "w") as f:
            f.write(f"Worst Class: {worst_class}\n")
            f.write(f"AUROC: {sorted_classes[0][1]:.4f}\n")
            f.write("Failure Cases (False Negatives - Lowest Anomaly Scores):\n")
            for i, fail in enumerate(top_failures):
                f.write(f"{i}. {fail['path']} (Score: {fail['img_score']:.4f})\n")
        
        return

    epoch_len = max(1, len(train_dataloader))
    for epoch in range(int(np.ceil(total_iters / epoch_len))):
        model.train()
        loss_list = []
        for img, label in train_dataloader:
            img = img.to(device)
            label = label.to(device)

            en, de = model(img)

            p_final = 0.7 if args.decoder in ('mamba', 'hybrid') else 0.9
            p = min(p_final * it / 3000, p_final)  # 3k iter 拉满
            loss = global_cosine_hm_percent(en, de, p=p, factor=0.1)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable.parameters(), max_norm=0.1)
            optimizer.step()
            loss_list.append(loss.item())
            lr_scheduler.step()

            # --- 定期评测 & 可视化 ---
            if (it + 1) % args.eval_every == 0:
                # Step-1: 选择图像级聚合器
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
                # else: 默认 max

                auroc_sp_list, ap_sp_list, f1_sp_list = [], [], []
                auroc_px_list, ap_px_list, f1_px_list, aupro_px_list = [], [], [], []

                for (item, test_data, calib_data) in zip(item_list, test_data_list, train_data_list):
                    test_dataloader = torch.utils.data.DataLoader(
                        test_data, batch_size=batch_size, shuffle=False, num_workers=args.workers
                    )

                    # -------- 常规评测 --------
                    results = evaluation_batch(
                        model, test_dataloader, device,
                        max_ratio=0.01, resize_mask=256,
                        aggregator=agg_fn,
                        flip_tta=args.flip_tta,
                        z_norm=args.z_norm,
                        z_calib_dataset=calib_data,
                        postproc=args.postproc,
                        topk_ratio=args.topk_ratio, gem_p=args.gem_p, fg_q=args.fg_q,
                        post_k=args.post_k, post_iters=args.post_iters,
                        rot_tta=args.rot_tta, ms_tta=args.ms_tta
                    )

                    auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results
                    auroc_sp_list.append(auroc_sp); ap_sp_list.append(ap_sp); f1_sp_list.append(f1_sp)
                    auroc_px_list.append(auroc_px); ap_px_list.append(ap_px); f1_px_list.append(f1_px); aupro_px_list.append(aupro_px)

                    print_fn('{}: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, '
                             'P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                                 item, auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px))

                    # -------- 可视化导出（若开启）--------
                    if args.viz_dir:
                        dump_root = os.path.join(args.viz_dir, item)
                        os.makedirs(dump_root, exist_ok=True)
                        vis_count = 0

                        # （可选）注册 decoder hooks，做早/晚层残差图
                        handles, layer_feats = [], []
                        if hasattr(model, "decoder") and register_decoder_hooks is not None:
                            try:
                                handles, layer_feats = register_decoder_hooks(model)
                            except Exception:
                                handles, layer_feats = [], []

                        # 逐样本可视化（最多 viz_n 张）
                        for batch in test_dataloader:
                            # 兼容 tuple/list/dict 批次
                            if isinstance(batch, (list, tuple)):
                                vimg = batch[0]
                            elif isinstance(batch, dict):
                                vimg = batch.get('image', batch.get('img', None))
                                if vimg is None:
                                    # 兜底取第一个键对应的值
                                    first_key = list(batch.keys())[0]
                                    vimg = batch[first_key]
                            else:
                                vimg = batch  # 兜底

                            vimg = vimg.to(device)
                            with torch.no_grad():
                                ven, vde = model(vimg)

                                # 组装两组 token （与 fuse_layer_decoder 对齐）
                                # —— 自适应构造两组（兼容：已分好两组 / 按层分组 / 其它长度）——
                                def _make_groups(ven_list, vde_list):
                                    # 1) 最常见：模型已经返回两组（低/高），直接取用
                                    if isinstance(ven_list, (list, tuple)) and isinstance(vde_list, (list, tuple)):
                                        if len(ven_list) == 2 and len(vde_list) == 2:
                                            E_low, E_high = _to_tokens(ven_list[0]), _to_tokens(ven_list[1])
                                            D_low, D_high = _to_tokens(vde_list[0]), _to_tokens(vde_list[1])
                                            return [E_low, E_high], [D_low, D_high]

                                        # 2) 返回了逐层列表（长度 >=4），按 encoder 的分组索引聚合
                                        if len(ven_list) >= 4 and len(vde_list) >= 4:
                                            enc_low_idx, enc_high_idx = fuse_layer_encoder[0], fuse_layer_encoder[1]
                                            E_low, D_low   = _group_mean_tokens(ven_list, vde_list, enc_low_idx)
                                            E_high, D_high = _group_mean_tokens(ven_list, vde_list, enc_high_idx)
                                            return [E_low, E_high], [D_low, D_high]

                                        # 3) 兜底：长度既不是 2 也 <4，就用首/尾两路近似
                                        E_low, E_high = _to_tokens(ven_list[0]), _to_tokens(ven_list[-1])
                                        D_low, D_high = _to_tokens(vde_list[0]), _to_tokens(vde_list[-1])
                                        return [E_low, E_high], [D_low, D_high]

                                    # 极端兜底：如果 ven/vde 不是 list/tuple，就把它们当成同一路
                                    E = _to_tokens(ven_list)
                                    D = _to_tokens(vde_list)
                                    return [E, E], [D, D]

                                en_groups, de_groups = _make_groups(ven, vde)


                                # 构造两组热力图 + 融合图（内部会裁掉 register tokens）
                                H_low, H_high, H_fused = build_g2r_maps(
                                    en_groups, de_groups, num_reg_tokens=num_reg_tokens, fuse="mean"
                                )
                                # HM-Percent: 前 p% 易点掩码（用于证明“易点降权”）
                                easy, hard, thr = percentile_easy_mask(H_fused, p=args.viz_p)

                            # 6列合并图
                            save_path = os.path.join(dump_root, f"qual6_{vis_count:03d}.png")
                            plot_qual6(vimg[0].detach().cpu(), H_low.cpu(), H_high.cpu(), H_fused.cpu(),
                                       bin_mask=(H_fused >= thr).cpu(), final_map=None, save_path=save_path)

                            # 六张拆分图
                            out_prefix = os.path.join(dump_root, f"qual6_{vis_count:03d}")
                            export_qual6_split(
                                img_chw=vimg[0].cpu(),
                                H_low=H_low.cpu(),
                                H_high=H_high.cpu(),
                                H_fused=H_fused.cpu(),
                                bin_mask=(H_fused >= thr).cpu(),   # 或传 None 用默认95%
                                final_map=None,                    # 若有“评测后最终热图”，在这里替换
                                out_prefix=out_prefix
                            )

                            # 易(蓝)/难(红) 叠加 —— 证明 HM-Percent 的聚焦作用
                            try:
                                img_np = vimg[0].detach().cpu().permute(1,2,0).numpy()
                                img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min() + 1e-6)
                                h = (H_fused[0] - H_fused.min()) / (H_fused.max() - H_fused.min() + 1e-6)
                                h = np.array(_PILImage.fromarray((h.cpu().numpy()*255).astype(np.uint8)).resize(
                                                (img_np.shape[1], img_np.shape[0]), resample=_PILImage.BILINEAR))/255.0
                                easy_up = np.array(_PILImage.fromarray(
                                    (easy[0].cpu().numpy().astype(np.uint8)*255)).resize(
                                    (img_np.shape[1], img_np.shape[0]), resample=_PILImage.NEAREST))>0
                                hard_up = ~easy_up
                                color = np.zeros_like(img_np)
                                color[easy_up] = np.array([0.2, 0.5, 1.0])  # 蓝=易
                                color[hard_up] = np.array([1.0, 0.2, 0.2])  # 红=难
                                mix = 0.6*img_np + 0.4*color
                                _PILImage.fromarray((mix*255).astype(np.uint8)).save(
                                    os.path.join(dump_root, f"easy_hard_{vis_count:03d}.png"))
                            except Exception:
                                pass

                            vis_count += 1
                            if vis_count >= args.viz_n:
                                break

                        # 绘制 ViT→Mamba 早/晚层残差条形图（每类 1 张）
                        if vis_count > 0 and handles and plot_layer_bars is not None:
                            try:
                                with torch.no_grad():
                                    _ = model(vimg)  # 用刚刚那个样本再跑一次以填充 layer_feats
                                bars = compute_layer_residual_bars(layer_feats, mode='delta')
                                select = tuple(int(s) for s in args.viz_layers.split(','))
                                plot_layer_bars(bars, select_layers=select,
                                                save_path=os.path.join(dump_root, "layer_bars.png"))
                            except Exception:
                                pass
                            finally:
                                for h in handles:
                                    try: h.remove()
                                    except Exception: pass

                print_fn('Mean: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, '
                        'P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                            np.mean(auroc_sp_list), np.mean(ap_sp_list), np.mean(f1_sp_list),
                            np.mean(auroc_px_list), np.mean(ap_px_list), np.mean(f1_px_list), np.mean(aupro_px_list)))
                model.train()

            it += 1
            if it == total_iters:
                break

            if (it + 1) % args.log_every == 0:
                print_fn('iter [{}/{}], loss:{:.4f}'.format(it, total_iters, np.mean(loss_list)))
                loss_list = []

    return

# ------------------------- 入口 -------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Few-shot Dinomaly with Mamba/Vit/Hybrid decoder (MVTEC)')
    # 数据路径与保存
    parser.add_argument('--data_path', type=str,
                        default='/sdb/huoyongzhen/Dinomaly/data/mvtec_anomaly_detection')
    parser.add_argument('--save_dir',  type=str,
                        default='/sdb/huoyongzhen/Dinomaly/data/final/mvtec/zhu/')
    parser.add_argument('--save_name', type=str, default='vitill_mvtec_uni_1shot')

    # few-shot & 设备 & 资源
    parser.add_argument('--shots', type=int, default=1, choices=[1, 2, 4],
                        help='few-shot per class')
    parser.add_argument('--cuda', type=int, default=0, help='CUDA device index')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--batch_size', type=int, default=4)

    # 解码器与深度
    parser.add_argument('--decoder', type=str, default='mamba',
                        choices=['vit', 'mamba', 'hybrid'],
                        help='decoder type: pure ViT / pure Mamba / hybrid (Mamba↔ViT)')
    parser.add_argument('--hybrid_pat', type=str, default='v,m,v,m,v,m,v,m',
                        help='pattern for hybrid across N blocks, m= Mamba, v= ViT')
    parser.add_argument('--decode_depth', type=int, default=8,
                        help='number of decoder blocks (default 8 to match your original)')

    # 编码器选择
    parser.add_argument('--encoder', type=str, default='dinov2reg_vit_base_14',
                        help='{dinov2reg_vit_small_14 | dinov2reg_vit_base_14 | dinov2reg_vit_large_14 | ...}')
    parser.add_argument("--bottleneck", type=str, default="geglu_dw",
                        choices=["none","bmlp","geglu_dw","gmlp_dw"])
    parser.add_argument("--bn_mlp_ratio", type=float, default=4.0)
    parser.add_argument("--bn_dw_kernel", type=int, default=5)
    parser.add_argument("--bn_drop", type=float, default=0.1)

    # 优化器 & 学习率调度
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--lr_end', type=float, default=2e-4)
    parser.add_argument('--warmup', type=int, default=100)

    # 日志与评测频率
    parser.add_argument('--eval_every', type=int, default=5000)
    parser.add_argument('--log_every',  type=int, default=100)

    # ===== Step-1 / Step-2 开关与超参（仅评测期生效） =====
    parser.add_argument('--hotspot_gem', action='store_true',
                        help='use Hotspot-GeM aggregator for image-level scores')
    parser.add_argument('--topk_ratio', type=float, default=0.02,
                        help='top-k ratio for hotspot selection, e.g., 0.02=2%')
    parser.add_argument('--gem_p', type=float, default=6.0,
                        help='GeM pooling p for image-level aggregation')
    parser.add_argument('--fg_q', type=int, default=70,
                        help='foreground gate quantile on token norm (0-100)')
    parser.add_argument('--flip_tta', action='store_true',
                        help='use horizontal flip TTA at eval time')
    parser.add_argument('--z_norm', action='store_true',
                        help='class-wise z-normalization for image scores at eval')
    parser.add_argument('--postproc', action='store_true',
                        help='morphological smoothing on heatmap at eval (Step-2)')
    parser.add_argument('--post_k', type=int, default=3, help='morph kernel size (odd)')
    parser.add_argument('--post_iters', type=int, default=1, help='morph iterations')
    parser.add_argument('--rot_tta', type=str, default='textures',
                        help="Rotation TTA policy: 'none'|'all'|'textures' or comma-separated class list")
    parser.add_argument('--ms_tta', type=str, default='0.75,1.0',
                        help="Multi-scale TTA factors, e.g. '0.75,1.0' (1.0 is implicit and will be skipped)")
    parser.add_argument('--tex_rot_tta', action='store_true',
                        help='Enable rotation TTA only for texture classes (carpet/grid/leather/tile/wood).')
    parser.add_argument('--i_agg', type=str, default='max', choices=['max', 'max_topk'],
                        help='Image-level aggregator: max (default) or max_topk')
    parser.add_argument('--i_alpha', type=float, default=0.6,
                        help='alpha for max_topk aggregator (0..1)')
    parser.add_argument('--i_top_percent', type=float, default=5.0,
                        help='top-percent for max_topk aggregator (e.g. 5 means top-5%% pixels)')

    # [ADDED] Checkpoint Loading
    parser.add_argument('--load_ckpt', type=str, default=None,
                        help='Path to load checkpoint from.')
    parser.add_argument('--visualize_only', action='store_true',
                        help='Skip training and run visualization/eval only.')
    parser.add_argument('--failure_case_mode', action='store_true',
                        help='Enable failure case analysis and WinCLIP-style visualization')

    # ====== 可视化选项 ======
    parser.add_argument('--viz_dir', type=str, default='./viz_out',
                        help='若非空，则在评测阶段导出每类若干张定性图')
    parser.add_argument('--viz_n', type=int, default=3, help='每类最多导出多少张样本')
    parser.add_argument('--viz_p', type=float, default=0.7, help='HM-Percent 的易点百分比 p')
    parser.add_argument('--viz_layers', type=str, default='2,6,8', help='绘制早/晚层残差的层编号')

    args = parser.parse_args()

    # 类别列表（MVTEC 15 类） - MODIFIED TO ONLY INCLUDE TRANSISTOR
    item_list = [
        'transistor'
    ]

    # Logger
    logger   = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info

    # 设备（按命令行选择）
    device = f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu'
    print_fn(device)

    # 训练
    train(item_list)
