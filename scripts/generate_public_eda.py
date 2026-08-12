from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import load_prices
from src.eda import generate_eda_figures
from src.features import build_feature_frame
from src.pairs import screen_pairs, select_top_pair
from src.splits import make_time_split
from src.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate EDA figures for a configured public/synthetic run.")
    parser.add_argument("--config", default="config/config_public.yaml")
    parser.add_argument("--require-public-data", action="store_true")
    parser.add_argument("--output-dir", default="outputs_public/m7_eda/figures")
    args = parser.parse_args()

    cfg = load_config(args.config)
    prices, summary = load_prices(cfg, require_public=args.require_public_data)
    split = make_time_split(prices.index, cfg)
    pair_table, _ = screen_pairs(prices.loc[split.train_index], cfg)
    asset_a, asset_b, alpha, beta, status = select_top_pair(pair_table)
    features = build_feature_frame(prices, asset_a, asset_b, cfg, alpha, beta)
    paths = generate_eda_figures(prices, features, Path(args.output_dir))
    print(f"EDA completed for {asset_a}-{asset_b} ({status}).")
    print(f"Data source: {summary.source}")
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
