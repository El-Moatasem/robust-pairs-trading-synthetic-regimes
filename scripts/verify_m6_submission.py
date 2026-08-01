from __future__ import annotations

from pathlib import Path


REQUIRED = [
    "README.md",
    "CONTRIBUTORS.md",
    "requirements.txt",
    "config/config.yaml",
    "config/config_public.yaml",
    "run_pipeline.py",
    "docs/METHODOLOGY.md",
    "docs/PEER_REVIEW_RESPONSE.md",
    "docs/RESULTS_AND_SIGNIFICANCE.md",
    "docs/M6_PROGRESS_AND_SCHEDULE.md",
    "tests/test_research_controls.py",
]

FORBIDDEN_DIRS = {".git", ".venv", ".idea", ".pytest_cache", "__pycache__"}


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    missing = [item for item in REQUIRED if not (root / item).exists()]
    forbidden = sorted({path.name for path in root.rglob("*") if path.is_dir() and path.name in FORBIDDEN_DIRS})
    if missing:
        raise SystemExit("Missing M6 files: " + ", ".join(missing))
    if forbidden:
        raise SystemExit("Remove generated/private directories before submission: " + ", ".join(forbidden))
    print("M6 source-package verification passed.")


if __name__ == "__main__":
    main()
