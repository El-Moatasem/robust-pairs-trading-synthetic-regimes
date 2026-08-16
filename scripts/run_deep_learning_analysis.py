from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import load_prices
from src.deep_learning import run_deep_fixed_split, run_deep_walk_forward, write_deep_outputs
from src.utils import load_config, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run DNN/GRU comparisons under the capstone research controls.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--require-public-data", action="store_true")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--walk-forward", action="store_true", help="Also run the expanding-window M7 analysis.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg["project"].get("random_seed", 42)))
    prices, summary = load_prices(cfg, require_public=args.require_public_data)
    if args.require_public_data and summary.is_synthetic:
        raise RuntimeError("Strict public-data run unexpectedly returned synthetic data.")
    output = args.output_dir or f"{cfg['outputs']['dir']}_deep"

    fixed, fixed_summary = run_deep_fixed_split(prices, cfg)
    walkforward = selected = None
    universality = None
    if args.walk_forward:
        walkforward, selected, universality = run_deep_walk_forward(prices, cfg)
    write_deep_outputs(output, fixed, fixed_summary, walkforward, selected, universality)

    print(f"Deep-learning analysis completed: {output}")
    print(f"Data source: {summary.source}")
    print(f"Synthetic data: {summary.is_synthetic}")
    print("Fixed-split validation-selected model:", fixed_summary.get("validation_selected_model"))
    if universality is not None:
        print("Universal ML/DL solution found:", universality["universal_solution_found"])
        print("Selected models by fold:")
        for row in universality["selected_models"]:
            print(" ", row)


if __name__ == "__main__":
    main()
