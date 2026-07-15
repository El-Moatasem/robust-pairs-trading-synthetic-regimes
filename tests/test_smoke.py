from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def test_pipeline_smoke() -> None:
    root = Path(__file__).resolve().parents[1]
    subprocess.check_call([sys.executable, str(root / "single_file_demo.py")], cwd=root)
    subprocess.check_call([sys.executable, str(root / "run_pipeline.py"), "--config", str(root / "config" / "config.yaml")], cwd=root)
