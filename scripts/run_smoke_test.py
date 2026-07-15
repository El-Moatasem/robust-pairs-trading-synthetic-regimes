from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    commands = [
        [sys.executable, str(root / "single_file_demo.py")],
        [sys.executable, str(root / "run_pipeline.py"), "--config", str(root / "config" / "config.yaml")],
    ]
    for cmd in commands:
        print("Running:", " ".join(cmd))
        subprocess.check_call(cmd, cwd=root)
    print("Smoke test passed.")


if __name__ == "__main__":
    main()
