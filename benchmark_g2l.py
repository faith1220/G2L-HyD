"""
Benchmark Script for Dinomaly Model (Table 4 Metrics)
======================================================

Measures the metrics reported in Dinomaly paper Table 4:
- Params (M): Trainable parameters in millions
- MACs (G): Multiply-Accumulate operations in billions
- Im/s: Images per second (throughput)

Usage:
    # Default: benchmark with dinov2reg_vit_base_14 encoder
    python benchmark_dinomaly.py

    # With specific encoder
    python benchmark_dinomaly.py --encoder dinov2reg_vit_small_14

    # Full configuration matching your training
    python benchmark_dinomaly.py --encoder dinov2reg_vit_base_14 \
        --decoder hybrid --hybrid_pat v,v,v,v,m,m,m,m \
        --bottleneck_variant gmlp_dw --decode_depth 8

    # Quick test
    python benchmark_dinomaly.py --runs 20 --warmup 5
"""

import os
import sys
import time
import argparse
import functools
import numpy as np

import torch
import torch.nn as nn
from tqdm import tqdm

# === Project Imports (same as test_mvtec.py) ===
from flops_profiler.profiler import FlopsProfiler
from g2l_modules.g2l_model import G2LHyD, DALRBlock
from g2l_modules import encoder
from g2l_modules.gfsr_blocks import (
    GFSRBlock, Attention, LinearAttention,
    DropoutSchedule, LinearGMLPTokenBlock, CGB, SAB,
    NoisyGEGLUBottleneck, MaskTokenBottleneck, HVQBottle, 
    RSCReconstructionHead, FeatureJitter
)


# ========== Helper Modules (same as test_mvtec.py) ==========
class ResidualScale(nn.Module):
    def __init__(self, module: nn.Module, init_scale: float = 0.2):
        super().__init__()
        self.module = module
        self.gamma = nn.Parameter(torch.tensor(init_scale, dtype=torch.float32))

    def forward(self, x, **kwargs):
        y = self.module(x, **kwargs)
        return x + self.gamma * (y - x)


class PostNorm(nn.Module):
    def __init__(self, module: nn.Module, dim: int, eps: float = 1e-6):
        super().__init__()
        self.module = module
        self.ln = nn.LayerNorm(dim, eps=eps)

    def forward(self, x, **kwargs):
        y = self.module(x, **kwargs)
        return self.ln(y)


# ========== Model Factory (same as test_mvtec.py) ==========
def make_vit_block(dim, heads, attn_type='linear'):
    attn_class = Attention if attn_type == 'full' else LinearAttention
    return GFSRBlock(
        dim=dim,
        num_heads=heads,
        mlp_ratio=4.,
        qkv_bias=True,
        norm_layer=functools.partial(nn.LayerNorm, eps=1e-8),
        attn=attn_class
    )


def make_mamba_block(dim, num_reg_tokens, args):
    def _parse_pair(s: str, default=(5, 7)):
        try:
            vals = tuple(int(x.strip()) for x in s.split(','))
            return vals if len(vals) == 2 else default
        except:
            return default

    ks_pair = _parse_pair(args.mamba_ks, default=(5, 7))
    dil_pair = _parse_pair(args.mamba_dilations, default=(1, 2))
    
    if args.mamba_dw_kernel > 0:
        ks_pair = (args.mamba_dw_kernel, args.mamba_dw_kernel)

    lss_kwargs = dict(
        ks=ks_pair,
        dilations=dil_pair,
        add_local3=bool(args.mamba_add_local3),
        add_asym=bool(args.mamba_add_asym),
        hss_learn_dir=True
    )

    blk = DALRBlock(
        embed_dim=dim,
        num_register_tokens=num_reg_tokens,
        num_hss=args.mamba_num_hss,
        scan_method=args.mamba_scan_method,
        num_scan_dirs=args.mamba_scan_dirs,
        lss_kwargs=lss_kwargs
    )
    blk = ResidualScale(blk, init_scale=0.2)
    blk = PostNorm(blk, dim)
    return blk


def make_decoder(kind, dim, heads, args, pattern=None, depth=8, num_reg_tokens=4):
    blocks = []
    
    if kind == 'vit':
        for _ in range(depth):
            blocks.append(make_vit_block(dim, heads, args.decoder_attn))
    
    elif kind == 'mamba':
        for _ in range(depth):
            blocks.append(make_mamba_block(dim, num_reg_tokens, args))
    
    else:  # hybrid
        pat = [s.strip().lower() for s in (pattern or 'v,m,v,m,v,m,v,m').split(',')]
        assert len(pat) == depth, f"Pattern length ({len(pat)}) != depth ({depth})"
        for p in pat:
            if p == 'm':
                blocks.append(make_mamba_block(dim, num_reg_tokens, args))
            elif p == 'v':
                blocks.append(make_vit_block(dim, heads, args.decoder_attn))
            else:
                raise ValueError(f"Unknown pattern: {p}")
    
    return nn.ModuleList(blocks)


def make_bottleneck(embed_dim, variant, args, num_reg_tokens=4):
    if variant == 'none':
        return nn.ModuleList([nn.Identity()])
    
    hf = int(2.67 * embed_dim)
    common = dict(drop=args.bn_drop, grad=0.7)
    
    if variant == 'cgb':
        return nn.ModuleList([CGB(embed_dim, hf, embed_dim, **common)])
    
    elif variant == 'gmlp_dw':
        return nn.ModuleList([SAB(
            in_features=embed_dim, 
            hidden_features=int(2 * embed_dim), 
            out_features=embed_dim,
            drop=args.bn_drop, 
            grad=0.7, 
            dw_kernel=args.bn_dw_kernel,
            num_register_tokens=num_reg_tokens, 
            has_cls=True
        )])
    
    elif variant == 'gmlp_lin':
        return nn.ModuleList([LinearGMLPTokenBlock(
            in_features=embed_dim, 
            hidden_features=int(2 * embed_dim), 
            out_features=embed_dim,
            drop=args.bn_drop, 
            grad=0.7,
            num_register_tokens=num_reg_tokens, 
            has_cls=True,
            token_mode='banded',
            bandwidth=7,
            rank=64
        )])
    
    elif variant == 'noisy_geglu':
        return nn.ModuleList([NoisyGEGLUBottleneck(
            in_features=embed_dim,
            hidden_features=int(8 * embed_dim / 3),
            out_features=embed_dim,
            dropout_schedule=DropoutSchedule(
                start_p=args.bn_drop_start, 
                end_p=args.bn_drop_end, 
                warmup_steps=args.bn_drop_warmup
            ),
            detach_ratio=0.0
        )])
    
    elif variant == 'jitter':
        return nn.ModuleList([FeatureJitter(sigma=args.jitter_sigma, p=1.0)])
    
    elif variant == 'mask':
        return nn.ModuleList([MaskTokenBottleneck(
            dim=embed_dim, 
            mask_ratio=args.mask_ratio, 
            learnable=True
        )])
    
    else:
        print(f"[WARN] Bottleneck {variant} not implemented, using Identity")
        return nn.ModuleList([nn.Identity()])


def build_model(args, device='cuda'):
    """Build the complete Dinomaly model."""
    # Encoder
    encoder_name = args.encoder
    encoder = encoder.load(encoder_name)
    
    # Determine embed dimensions
    if 'small' in encoder_name:
        embed_dim, num_heads = 384, 6
    elif 'base' in encoder_name:
        embed_dim, num_heads = 768, 12
    elif 'large' in encoder_name:
        embed_dim, num_heads = 1024, 16
    else:
        raise RuntimeError(f"Unknown encoder architecture: {encoder_name}")
    
    num_reg_tokens = 4
    target_layers = [2, 3, 4, 5, 6, 7, 8, 9]
    fuse_layer_encoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    fuse_layer_decoder = [[0, 1, 2, 3], [4, 5, 6, 7]]
    
    # Bottleneck
    bottleneck = make_bottleneck(embed_dim, args.bottleneck_variant, args, num_reg_tokens)
    
    # Decoder
    decoder = make_decoder(
        args.decoder, embed_dim, num_heads, args,
        pattern=args.hybrid_pat, 
        depth=args.decode_depth,
        num_reg_tokens=num_reg_tokens
    )
    
    # Full model
    model = G2LHyD(
        encoder=encoder,
        bottleneck=bottleneck,
        decoder=decoder,
        target_layers=target_layers,
        mask_neighbor_size=0,
        fuse_layer_encoder=fuse_layer_encoder,
        fuse_layer_decoder=fuse_layer_decoder
    )
    
    # Freeze encoder (same as training)
    for param in model.encoder.parameters():
        param.requires_grad = False
    
    return model.to(device).eval()


# ========== Benchmark Functions ==========
def count_parameters(model: nn.Module, only_trainable: bool = True) -> int:
    if only_trainable:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def profile_flops_macs(model, input_shape, device='cuda', print_profile=False):
    """Profile FLOPs and MACs using FlopsProfiler."""
    prof = FlopsProfiler(model)
    x = torch.randn(*input_shape, device=device)
    
    # Warmup
    with torch.no_grad():
        _ = model(x)
    
    prof.start_profile()
    with torch.no_grad():
        _ = model(x)
    
    flops = prof.get_total_flops()
    macs = prof.get_total_macs()
    params = prof.get_total_params()
    
    if print_profile:
        prof.print_model_profile(profile_step=1, detailed=False)
    
    prof.end_profile()
    
    return {
        'flops': flops,
        'flops_G': flops / 1e9,
        'macs': macs,
        'macs_G': macs / 1e9,
        'params': params,
        'params_M': params / 1e6,
    }


def measure_throughput(model, input_shape, device='cuda', warmup=10, runs=100):
    """Measure inference throughput (Im/s)."""
    model.eval()
    batch_size = input_shape[0]
    x = torch.randn(*input_shape, device=device)
    
    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(x)
    
    if 'cuda' in device:
        torch.cuda.synchronize()
    
    # Measure
    latencies = []
    with torch.no_grad():
        for _ in tqdm(range(runs), desc="Measuring throughput"):
            if 'cuda' in device:
                torch.cuda.synchronize()
            
            start = time.perf_counter()
            _ = model(x)
            
            if 'cuda' in device:
                torch.cuda.synchronize()
            
            end = time.perf_counter()
            latencies.append((end - start) * 1000)
    
    latency_ms = np.mean(latencies)
    latency_std = np.std(latencies)
    throughput = (batch_size / latency_ms) * 1000
    
    return {
        'latency_ms': latency_ms,
        'latency_std_ms': latency_std,
        'throughput_ips': throughput,
    }


# ========== Main Benchmark ==========
def run_benchmark(args):
    device = f'cuda:{args.cuda}' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")
    if device.startswith('cuda'):
        print(f"GPU: {torch.cuda.get_device_name(args.cuda)}")
    
    # Build model
    print("\nBuilding model...")
    print(f"  Encoder: {args.encoder}")
    print(f"  Decoder: {args.decoder} (depth={args.decode_depth})")
    if args.decoder == 'hybrid':
        print(f"  Pattern: {args.hybrid_pat}")
    print(f"  Bottleneck: {args.bottleneck_variant}")
    
    model = build_model(args, device)
    
    # Count parameters
    total_params = count_parameters(model, only_trainable=False)
    trainable_params = count_parameters(model, only_trainable=True)
    
    # Input shape
    input_shape = (args.batch_size, 3, args.img_size, args.img_size)
    print(f"\nInput shape: {input_shape}")
    
    # Profile FLOPs/MACs
    print("\nProfiling FLOPs and MACs...")
    profile_results = profile_flops_macs(model, input_shape, device, args.detailed)
    
    # Measure throughput
    print(f"\nMeasuring throughput (warmup={args.warmup}, runs={args.runs})...")
    throughput_results = measure_throughput(model, input_shape, device, args.warmup, args.runs)
    
    # Combine results
    results = {
        'total_params': total_params,
        'total_params_M': total_params / 1e6,
        'trainable_params': trainable_params,
        'trainable_params_M': trainable_params / 1e6,
        **profile_results,
        **throughput_results,
    }
    
    # Print results
    print("\n" + "=" * 70)
    print("BENCHMARK RESULTS (Table 4 Style)")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Encoder:    {args.encoder}")
    print(f"  Decoder:    {args.decoder} ({args.hybrid_pat if args.decoder=='hybrid' else 'pure'})")
    print(f"  Bottleneck: {args.bottleneck_variant}")
    print(f"  Input Size: {args.img_size}x{args.img_size}")
    print(f"  Batch Size: {args.batch_size}")
    
    print(f"\nParameters:")
    print(f"  Total:      {results['total_params_M']:.2f} M")
    print(f"  Trainable:  {results['trainable_params_M']:.2f} M")
    
    print(f"\nComputational Cost:")
    print(f"  MACs:       {results['macs_G']:.2f} G")
    print(f"  FLOPs:      {results['flops_G']:.2f} G")
    
    print(f"\nThroughput:")
    print(f"  Latency:    {results['latency_ms']:.2f} ± {results['latency_std_ms']:.2f} ms")
    print(f"  Im/s:       {results['throughput_ips']:.2f}")
    
    print("\n" + "-" * 70)
    print("Table 4 Format:")
    print("-" * 70)
    print(f"| {'Method':<20} | {'Params (M)':<12} | {'MACs (G)':<12} | {'Im/s':<10} |")
    print(f"| {'-'*20} | {'-'*12} | {'-'*12} | {'-'*10} |")
    print(f"| {'Ours':<20} | {results['trainable_params_M']:>10.2f} | {results['macs_G']:>10.2f} | {results['throughput_ips']:>8.2f} |")
    print("=" * 70)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark Dinomaly Model for Table 4")
    
    # Device
    parser.add_argument('--cuda', type=int, default=0)
    
    # Model config (match test_mvtec.py defaults)
    parser.add_argument('--encoder', type=str, default='dinov2reg_vit_base_14')
    parser.add_argument('--decoder', type=str, default='hybrid', choices=['vit', 'mamba', 'hybrid'])
    parser.add_argument('--decoder_attn', type=str, default='linear', choices=['linear', 'full'])
    parser.add_argument('--hybrid_pat', type=str, default='v,v,v,v,m,m,m,m')
    parser.add_argument('--decode_depth', type=int, default=8)
    parser.add_argument('--bottleneck_variant', type=str, default='gmlp_dw')
    
    # Bottleneck args
    parser.add_argument('--bn_drop', type=float, default=0.1)
    parser.add_argument('--bn_dw_kernel', type=int, default=5)
    parser.add_argument('--bn_drop_start', type=float, default=0.0)
    parser.add_argument('--bn_drop_end', type=float, default=0.2)
    parser.add_argument('--bn_drop_warmup', type=int, default=1000)
    parser.add_argument('--jitter_sigma', type=float, default=0.10)
    parser.add_argument('--mask_ratio', type=float, default=0.30)
    
    # Mamba args
    parser.add_argument('--mamba_ks', type=str, default='5,7')
    parser.add_argument('--mamba_dilations', type=str, default='1,2')
    parser.add_argument('--mamba_dw_kernel', type=int, default=0)
    parser.add_argument('--mamba_add_local3', type=int, default=1)
    parser.add_argument('--mamba_add_asym', type=int, default=1)
    parser.add_argument('--mamba_num_hss', type=int, default=3)
    parser.add_argument('--mamba_scan_method', type=str, default='hilbert')
    parser.add_argument('--mamba_scan_dirs', type=int, default=8)
    
    # Benchmark config
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for benchmark')
    parser.add_argument('--img_size', type=int, default=448, help='Input image size (448 for training)')
    parser.add_argument('--warmup', type=int, default=10, help='Warmup iterations')
    parser.add_argument('--runs', type=int, default=100, help='Benchmark iterations')
    parser.add_argument('--detailed', action='store_true', help='Print detailed profile')
    
    args = parser.parse_args()
    run_benchmark(args)
