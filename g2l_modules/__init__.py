# G2L-HyD/g2l_modules/__init__.py — GFSR & model factory
from g2l_modules.gfsr_blocks import GFSRBlock
from g2l_modules.g2l_model import DALRBlock


def make_gfsr_block(embed_dim: int):
    """Create a GFSR (Global Feature-Space Reconstruction) block."""
    return GFSRBlock(dim=embed_dim)


def make_dalr_block(embed_dim: int):
    """Create a DALR (Direction-Aware Local Refinement) block."""
    return DALRBlock(embed_dim=embed_dim, num_register_tokens=0)
