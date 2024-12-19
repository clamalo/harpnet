"""
HARPNET precipitation downscaling project package.

This package contains:
- Data preparation code (interpolation, tiling, etc.)
- Model definitions and architectures (UNet with attention)
- Training and evaluation code
- Utility functions (plotting, logging, cleanup)
"""

__all__ = [
    "config",
    "utils",
    "data",
    "models",
    "training"
]