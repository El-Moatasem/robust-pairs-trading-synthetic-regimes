from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic_spreads(base_spread: pd.Series, cfg: dict) -> pd.DataFrame:
    scfg = cfg["synthetic"]
    rng = np.random.default_rng(int(scfg.get("seed", 42)))
    n = int(scfg["scenario_length"])
    base_std = float(base_spread.diff().std()) or 0.01
    base_level = float(base_spread.mean())
    scenarios = {}
    regimes = list(scfg["regimes"].items())
    for i in range(int(scfg["n_scenarios"])):
        regime_name, params = regimes[i % len(regimes)]
        vol = base_std * float(params["vol_scale"])
        kappa = 0.08 * float(params["mean_reversion_scale"])
        jump_prob = float(params["jump_prob"])
        corr_break_prob = float(params["corr_break_prob"])
        s = np.zeros(n)
        s[0] = rng.normal(base_level, base_std)
        drift_shift = 0.0
        for t in range(1, n):
            if rng.random() < corr_break_prob:
                drift_shift += rng.normal(0, 0.5 * base_std)
            shock = rng.normal(0, vol)
            if rng.random() < jump_prob:
                shock += rng.normal(0, 5 * vol)
            s[t] = s[t - 1] + kappa * (base_level + drift_shift - s[t - 1]) + shock
        scenarios[f"scenario_{i+1:03d}_{regime_name}"] = s
    idx = pd.RangeIndex(n, name="step")
    return pd.DataFrame(scenarios, index=idx)


def summarize_scenarios(scenarios: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in scenarios.columns:
        s = scenarios[col]
        rows.append({
            "scenario": col,
            "std": float(s.std()),
            "max_abs_spread": float(s.abs().max()),
            "autocorr_1": float(s.autocorr(1)),
        })
    return pd.DataFrame(rows)
