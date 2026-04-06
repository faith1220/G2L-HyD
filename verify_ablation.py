
import torch
import torch.nn as nn
from g2l_modules.g2l_model import G2LHyD, DALRBlock
from g2l_modules.gfsr_blocks import SAB

# Mock arguments
class Args:
    def __init__(self):
        self.mamba_ks = '5,7'
        self.mamba_dilations = '1,2'
        self.mamba_add_local3 = True
        self.mamba_add_asym = True
        self.mamba_num_hss = 3
        self.mamba_scan_method = 'hilbert'
        self.mamba_scan_dirs = 8
        self.mamba_dw_kernel = 0
        self.decoder_attn = 'linear'
        self.decoder = 'mamba'
        self.hybrid_pat = None
        self.decode_depth = 8
        self.bottleneck_variant = 'gmlp_dw'
        self.bn_drop = 0.0
        self.bn_dw_kernel = 5
        self.mask_ratio = 0.0
        self.jitter_sigma = 0.0
        self.vq_k = 1024
        self.vq_beta = 0.25
        self.vq_ema_decay = 0.99
        self.bn_drop_start = 0.0
        self.bn_drop_end = 0.0
        self.bn_drop_warmup = 0

args = Args()

# Mock Encoder
class MockEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.num_register_tokens = 4
        self.embed_dim = 384
        self.blocks = nn.ModuleList([nn.Identity() for _ in range(12)])
    def prepare_tokens(self, x): return x

def make_model(config_name):
    print(f"Testing configuration: {config_name}")
    
    # Reset args to default
    args.mamba_ks = '5,7'
    args.mamba_add_local3 = True
    args.mamba_add_asym = True
    args.mamba_num_hss = 3
    args.bottleneck_variant = 'gmlp_dw'
    
    if config_name == 'Baseline + GFSR only':
        args.bottleneck_variant = 'none' # No SAB
        args.mamba_ks = '0,0' # Try to disable DALR?
        args.mamba_add_local3 = False
        args.mamba_add_asym = False
        # Note: '0,0' might fail if not handled
        
    elif config_name == 'Baseline + DALR only':
        args.bottleneck_variant = 'none'
        args.mamba_num_hss = 0 # Disable GFSR
        
    elif config_name == 'Full':
        pass # Defaults
        
    # ... instantiation logic copied/adapted from test_visa.py ...
    # Simplified for verification
    
    def _parse_pair(s: str, default=(5, 7)):
        try:
            vals = tuple(int(x.strip()) for x in s.split(','))
            if len(vals) != 2: return default
            return vals
        except Exception: return default

    ks_pair = _parse_pair(args.mamba_ks, default=(5, 7))
    # Check if we can disable DALR
    if config_name == 'Baseline + GFSR only':
        # If we want to disable DALR, we need ks to be empty or handled
        pass

    print(f"  ks_pair: {ks_pair}")
    print(f"  num_hss: {args.mamba_num_hss}")
    print(f"  bottleneck: {args.bottleneck_variant}")

    try:
        # Try to instantiate DALRBlock
        lss_kwargs = dict(
            ks=ks_pair,
            dilations=(1,2),
            add_local3=args.mamba_add_local3,
            add_asym=args.mamba_add_asym,
            hss_learn_dir=True
        )
        blk = DALRBlock(
            embed_dim=384,
            num_register_tokens=4,
            num_hss=int(args.mamba_num_hss),
            lss_kwargs=lss_kwargs
        )
        print("  DALRBlock instantiated successfully.")
        
        # Check internal structure
        print(f"  DALRLayerBlock locals: {len(blk.lss.locals)}")
        print(f"  DALRLayerBlock hss: {len(blk.lss.hss)}")
        
    except Exception as e:
        print(f"  Failed to instantiate: {e}")

make_model('Baseline + GFSR only')
make_model('Baseline + DALR only')
make_model('Full')
