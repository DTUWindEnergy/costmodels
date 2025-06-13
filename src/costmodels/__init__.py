import tempfile
from pathlib import Path

import jax

jax_cache_dir = Path(tempfile.gettempdir()) / "jax_cache"
jax.config.update("jax_compilation_cache_dir", str(jax_cache_dir))
jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
jax.config.update("jax_enable_x64", True)
