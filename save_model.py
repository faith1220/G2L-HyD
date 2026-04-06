import os
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"

import warnings
warnings.filterwarnings("ignore")

import sys
import shutil
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import numpy as np
import random
import argparse
import logging

from functools import partial
from torch.utils.data import DataLoader, ConcatDataset
from torchvision.datasets import ImageFolder
from sklearn.metrics import roc_auc_score, average_precision_score

# ====== 项目内模块 ======
from dataset import get_data_transforms, get_strong_transforms
from dataset import MVTecDataset
from g2l_modules.g2l_model import G2LHyD, G2LHyDv2, DALRBlock   # ★ 引入 Mamba 解码块
from g2l_modules import encoder
from dinov1.utils import trunc_normal_
from g2l_modules.gfsr_blocks import (
    GFSRBlock, bMlp, Attention, LinearAttention, LinearAttention,
    ConvBlock, FeatureJitter, CGB
)
from utils import (
    evaluation_batch, global_cosine, regional_cosine_hm_percent,
    global_cosine_hm_percent, WarmCosineScheduler,

    # ★ 你已将这些函数贴到 utils.py 顶部，这里直接导入使用
    compute_image_score_from_heatmap
)
from ptflops import get_model_complexity_info
from optimizers import StableAdamW
import copy
import itertools
import functools

# DDP / 多进程可用性
try:
    import torch.distributed as dist
except Exception:
    dist = None


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
    用一个可学习的 gamma（默认0.2）把子模块的改变量“压一压”，防止一上来破坏分布。
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
    子模块输出后做一个 LayerNorm（eps更小），稳住数值范围。
    """
    def __init__(self, module: nn.Module, dim: int, eps: float = 1e-6):
        super().__init__()
        self.module = module
        self.ln = nn.LayerNorm(dim, eps=eps)

    def forward(self, x, **kwargs):
        y = self.module(x, **kwargs)
        return self.ln(y)

def is_primary_process():
    """仅在主进程/单卡返回 True，用于避免 DDP 多进程同时写文件。"""
    if dist is None or not dist.is_available() or not dist.is_initialized():
        return True
    try:
        return dist.get_rank() == 0
    except Exception:
        return True

def safe_torch_save(state, path, print_fn=print):
    """带目录创建、磁盘余量提示、错误回显到 stderr 的保存函数。"""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        # 简单空间检查（留 200MB 余量）
        usage = shutil.disk_usage(os.path.dirname(path))
        if usage.free < 200 * (1024 ** 2):
            print_fn(f"[ckpt][WARN] Disk free {usage.free/1e9:.2f} GB < 0.2 GB, may fail.")
        torch.save(state, path)
        print_fn(f"[ckpt] saved to: {path}")
    except Exception as e:
        msg = f"[ckpt][ERROR] save failed at {path}: {repr(e)}"
        print_fn(msg)
        print(msg, file=sys.stderr)

# ------------------------- Checkpoint 保存（封装） -------------------------
def save_checkpoint(path, model, optimizer, lr_scheduler, it, args, print_fn=print):
    """
    在给定路径保存训练中间状态（包含模型、优化器、调度器与随机数状态）。
    """
    state = {
        "iter": it,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "lr_scheduler": lr_scheduler.state_dict(),
        "args": vars(args),
        "rng_torch": torch.get_rng_state(),
        "rng_np": np.random.get_state(),
        "rng_py": random.getstate(),
    }
    if torch.cuda.is_available():
        state["rng_cuda"] = torch.cuda.get_rng_state_all()
    safe_torch_save(state, path, print_fn=print_fn)

# ------------------------- 训练主函数 -------------------------
def train(item_list, print_fn):
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
        rng = random.Random(1 + i)  # 固定随机，按类偏移
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

    # ★ 按你的要求，reg 固定为 4
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
                blk = DALRBlock(embed_dim=dim, num_register_tokens=num_reg_tokens)
                blk = ResidualScale(blk, init_scale=0.2)   # ★ 残差缩放
                blk = PostNorm(blk, dim)                   # ★ 后归一化
                blocks.append(blk)
        else:  # hybrid
            pat = [s.strip().lower() for s in (pattern or 'v,m,v,m,v,m,v,m').split(',')]
            assert len(pat) == depth
            for p in pat:
                if p == 'm':
                    blk = DALRBlock(embed_dim=dim, num_register_tokens=num_reg_tokens)
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
            hidden_features=int(2.67 * embed_dim),  # ≈ 2.67x 宽度
            out_features=embed_dim,
            drop=0.2,
            grad=0.7)                                # 软阻断梯度，few-shot 更稳
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
        lr = lr * 0.5            # → 1e-3，避免过更新
        warmup = max(warmup, 500)  # 更长热身，减小早期漂移
    optimizer = StableAdamW([{'params': trainable.parameters()}],
                            lr=lr, betas=(0.9, 0.999), weight_decay=1e-4, amsgrad=True, eps=1e-10)
    lr_scheduler = WarmCosineScheduler(
        optimizer, base_value=lr, final_value=args.lr_end, total_iters=total_iters, warmup_iters=warmup
    )

    print_fn('train image number: {}'.format(len(train_data)))
    print_fn(f'few-shot setting: {shots_per_class} per class')

    # ====== 保存相关状态 ======
    saved_at_threshold = False

    it = 0
    epoch_len = max(1, len(train_dataloader))
    for epoch in range(int(np.ceil(total_iters / epoch_len))):
        model.train()
        loss_list = []
        for img, label in train_dataloader:
            img = img.to(device)
            label = label.to(device)

            en, de = model(img)

            p_final = 0.7 if args.decoder in ('mamba', 'hybrid') else 0.9
            p = min(p_final * it / 3000, p_final)  # 3k iter 再拉满

            loss = global_cosine_hm_percent(en, de, p=p, factor=0.1)

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(trainable.parameters(), max_norm=0.1)
            optimizer.step()
            loss_list.append(loss.item())
            lr_scheduler.step()

            # --- 定期评测 ---
            if (it + 1) % args.eval_every == 0:
                # Step-1：若启用 Hotspot-GeM，则构造聚合器
                # Step-1：选择图像级聚合器
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
                # else: 使用默认 max（evaluation_batch 内部会走 max 聚合）

                auroc_sp_list, ap_sp_list, f1_sp_list = [], [], []
                auroc_px_list, ap_px_list, f1_px_list, aupro_px_list = [], [], [], []

                # 注意：把 few-shot 的该类训练数据（K 张正常图）传给 z-norm 校准
                for (item, test_data, calib_data) in zip(item_list, test_data_list, train_data_list):
                    test_dataloader = torch.utils.data.DataLoader(
                        test_data, batch_size=batch_size, shuffle=False, num_workers=args.workers
                    )

                    results = evaluation_batch(
                        model, test_dataloader, device,
                        max_ratio=0.01, resize_mask=256,
                        aggregator=agg_fn,               # Step-1 Hotspot-GeM
                        flip_tta=args.flip_tta,          # Step-1 Flip-TTA
                        z_norm=args.z_norm,              # Step-1 z-norm
                        z_calib_dataset=calib_data,      # few-shot 正常图用于 μ/σ
                        postproc=args.postproc,          # Step-2 像素后处理
                        topk_ratio=args.topk_ratio, gem_p=args.gem_p, fg_q=args.fg_q,
                        post_k=args.post_k, post_iters=args.post_iters,   # <<< 新增两参
                        rot_tta=args.rot_tta, ms_tta=args.ms_tta          # <<< 旋转TTA和多尺度TTA
                    )

                    auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px = results
                    auroc_sp_list.append(auroc_sp); ap_sp_list.append(ap_sp); f1_sp_list.append(f1_sp)
                    auroc_px_list.append(auroc_px); ap_px_list.append(ap_px); f1_px_list.append(f1_px); aupro_px_list.append(aupro_px)

                    print_fn('{}: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, '
                             'P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                                item, auroc_sp, ap_sp, f1_sp, auroc_px, ap_px, f1_px, aupro_px))

                print_fn('Mean: I-Auroc:{:.4f}, I-AP:{:.4f}, I-F1:{:.4f}, '
                         'P-AUROC:{:.4f}, P-AP:{:.4f}, P-F1:{:.4f}, P-AUPRO:{:.4f}'.format(
                            np.mean(auroc_sp_list), np.mean(ap_sp_list), np.mean(f1_sp_list),
                            np.mean(auroc_px_list), np.mean(ap_px_list), np.mean(f1_px_list), np.mean(aupro_px_list)))
                model.train()

            # ====== 计数与保存 ======
            it += 1

            # （1）阈值保存：达到/超过 checkpoint_at 才保存，且仅一次；DDP 仅 rank0
            if (not saved_at_threshold) and (it >= args.checkpoint_at) and is_primary_process():
                ckpt_dir = os.path.join(args.save_dir, args.save_name, "ckpts")
                ckpt_path = os.path.join(ckpt_dir, f"iter_{args.checkpoint_at:06d}.pth")
                save_checkpoint(ckpt_path, model, optimizer, lr_scheduler, it, args, print_fn=print_fn)
                saved_at_threshold = True

            # （2）周期性保存（可选，默认关闭；注意磁盘空间）
            if args.checkpoint_every > 0 and is_primary_process() and (it % args.checkpoint_every == 0):
                ckpt_dir = os.path.join(args.save_dir, args.save_name, "ckpts")
                ckpt_path = os.path.join(ckpt_dir, f"iter_{it:06d}.pth")
                save_checkpoint(ckpt_path, model, optimizer, lr_scheduler, it, args, print_fn=print_fn)

            if it >= total_iters:
                break

            if (it + 1) % args.log_every == 0:
                print_fn('iter [{}/{}], loss:{:.4f}'.format(it, total_iters, np.mean(loss_list)))
                loss_list = []

    # ===== 训练结束后总是保存一个 final（含最后 iter）=====
    if is_primary_process():
        final_dir = os.path.join(args.save_dir, args.save_name, "ckpts")
        final_path = os.path.join(final_dir, f"final_iter_{it:06d}.pth")
        save_checkpoint(final_path, model, optimizer, lr_scheduler, it, args, print_fn=print_fn)

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

    # ===== Checkpoint 相关参数（新增） =====
    parser.add_argument('--checkpoint_at', type=int, default=20000,
                        help='当迭代 >= 该值时保存一次（仅一次）')
    parser.add_argument('--checkpoint_every', type=int, default=0,
                        help='若 >0，则每 N 次迭代保存一次（仅主进程）')

    args = parser.parse_args()

    # 类别列表（MVTEC 15 类）
    item_list = [
        'carpet', 'grid', 'leather', 'tile', 'wood',
        'bottle', 'cable', 'capsule', 'hazelnut', 'metal_nut',
        'pill', 'screw', 'toothbrush', 'transistor', 'zipper'
    ]

    # Logger
    logger   = get_logger(args.save_name, os.path.join(args.save_dir, args.save_name))
    print_fn = logger.info

    # 设备（按命令行选择）
    device = f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu'
    print_fn(device)

    # 训练
    train(item_list, print_fn)
