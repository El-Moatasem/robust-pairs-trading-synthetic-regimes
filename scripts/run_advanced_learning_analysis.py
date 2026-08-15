from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.advanced_learning import run_advanced_fixed_split, run_advanced_walk_forward, write_advanced_outputs
from src.data import load_prices
from src.utils import load_config, set_seed


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LSTM, TCN, regime-aware MoE and uncertainty-consensus experiments.")
    parser.add_argument("--config", default="config/config_m7_public.yaml")
    parser.add_argument("--output-dir", default="outputs_advanced_m7_public")
    parser.add_argument("--require-public-data", action="store_true")
    parser.add_argument("--walk-forward", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    set_seed(int(cfg.get("project", {}).get("random_seed", 42)))
    prices, metadata = load_prices(cfg, require_public=args.require_public_data)
    print(f"Data source: {metadata.source}")
    print(f"Synthetic data: {metadata.is_synthetic}")
    if args.require_public_data and bool(metadata.is_synthetic):
        raise RuntimeError("Strict public-data run unexpectedly received synthetic data.")

    if args.walk_forward:
        wf, selected, gated, universality, regimes = run_advanced_walk_forward(prices, cfg)
        write_advanced_outputs(args.output_dir, walkforward=wf, selected=selected, gated_summary=gated, universality=universality, regimes=regimes)
        print("\nValidation-selected advanced policy by fold")
        print(selected[["fold", "pair", "policy", "validation_f1", "pnl", "sharpe", "pair_passes_all"]].to_string(index=False))
        print("\nStrict statistical-gate policy summary")
        print(gated.to_string(index=False))
        print("\nStrict universal solution found:", universality["strict_universal_solution_found"])
        print("Observed gated-robust policies:", universality["observed_gated_robust_policies"])
    else:
        fixed, summary = run_advanced_fixed_split(prices, cfg)
        write_advanced_outputs(args.output_dir, fixed=fixed, fixed_summary=summary)
        print("\nAdvanced fixed-split results")
        print(fixed.to_string(index=False))
        print(summary)


if __name__ == "__main__":
    main()
