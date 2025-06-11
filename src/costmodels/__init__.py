import tempfile
from pathlib import Path

import jax

from costmodels.battery_cost import BatteryCostModel
from costmodels.dtu_offshore import DTUOffshoreCostModel
from costmodels.minimalistic import MinimalisticCostModel
from costmodels.nrel import NRELCostModel
from costmodels.pv import PVCostModel

__all__ = [
    "DTUOffshoreCostModel",
    "MinimalisticCostModel",
    "NRELCostModel",
    "PVCostModel",
    "BatteryCostModel",
]

jax_cache_dir = Path(tempfile.gettempdir()) / "jax_cache"
jax.config.update("jax_compilation_cache_dir", str(jax_cache_dir))
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_enable_x64", True)
