import glob
import subprocess
import sys
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).resolve().parents[1] / "examples"
scripts = glob.glob(str(EXAMPLES_DIR / "*.py"))
exclude = ["topfarm_integration.py"]


@pytest.mark.parametrize("script", [s for s in scripts if Path(s).name not in exclude])
def test_example_runs(script):
    subprocess.run([sys.executable, str(EXAMPLES_DIR / script)], check=True)
