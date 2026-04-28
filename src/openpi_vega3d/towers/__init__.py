"""Generative vision tower registry.

Usage:
    from openpi_vega3d.towers import TOWER_REGISTRY
    tower_cls = TOWER_REGISTRY["vae"]
    tower = tower_cls(checkpoint_dir="path/to/sd21")

Tower classes are lazily imported to avoid pulling in heavy dependencies
(diffusers, einops, easydict) at module load time.
"""

from openpi_vega3d.towers.base import BaseTower


class _LazyRegistry(dict):
    """Dict that lazily imports tower classes on first access."""

    _TOWER_MAP = {
        "vae": ("openpi_vega3d.towers.vae_tower", "VAETower"),
        "wan_t2v": ("openpi_vega3d.towers.wan_tower", "WanT2VTower"),
    }

    def __missing__(self, key):
        if key not in self._TOWER_MAP:
            raise KeyError(f"Unknown tower: {key!r}. Available: {list(self._TOWER_MAP)}")
        module_path, class_name = self._TOWER_MAP[key]
        import importlib
        mod = importlib.import_module(module_path)
        cls = getattr(mod, class_name)
        self[key] = cls
        return cls

    def __contains__(self, key):
        return key in self._TOWER_MAP or super().__contains__(key)

    def keys(self):
        return self._TOWER_MAP.keys()

    def __repr__(self):
        return f"TOWER_REGISTRY({list(self._TOWER_MAP.keys())})"


TOWER_REGISTRY: dict[str, type[BaseTower]] = _LazyRegistry()
