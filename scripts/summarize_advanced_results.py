from __future__ import annotations

import argparse
import json
from pathlib import Path
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize LSTM/TCN/regime/uncertainty experiments.")
    parser.add_argument("--outputs", default="outputs_advanced_m7_public")
    args = parser.parse_args()
    tables = Path(args.outputs) / "tables"
    comparison = pd.read_csv(tables / "advanced_walk_forward_policy_comparison.csv")
    selected = pd.read_csv(tables / "advanced_validation_selected_walk_forward.csv")
    gated = pd.read_csv(tables / "advanced_strict_gate_policy_summary.csv")
    passed = pd.read_csv(tables / "advanced_passed_and_profitable.csv")
    universality = json.loads((tables / "advanced_universality_summary.json").read_text())

    print("\nValidation-selected advanced policy by fold")
    print(selected[["fold", "pair", "policy", "validation_f1", "pnl", "sharpe", "pair_passes_all"]].to_string(index=False))
    print("\nPassed-and-profitable advanced cases")
    if passed.empty:
        print("None")
    else:
        print(passed[["fold", "pair", "policy", "pnl", "sharpe", "validation_f1"]].to_string(index=False))
    print("\nStrict statistical-gate fixed-policy summary")
    print(gated.to_string(index=False))
    print("\nStrict universal solution found:", universality["strict_universal_solution_found"])
    print("Observed gated-robust policies:", universality["observed_gated_robust_policies"])
    print("\nAll policies/folds")
    print(comparison[["fold", "pair", "pair_passes_all", "policy", "validation_f1", "pnl", "sharpe", "strict_gate_pnl"]].to_string(index=False))


if __name__ == "__main__":
    main()
