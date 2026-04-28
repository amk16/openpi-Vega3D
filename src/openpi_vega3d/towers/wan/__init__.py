"""WAN model components for the generative tower.

Trimmed from the upstream VEGA-3D wan/ package: only configs and modules
are imported. Pipeline classes (WanT2V, WanI2V, WanFLF2V, WanVace) are
excluded to avoid pulling in unnecessary dependencies.
"""

from . import configs, modules
