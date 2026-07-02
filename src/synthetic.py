from __future__ import annotations

import numpy as np
import pandas as pd


def generate_synthetic_spread(
    n_steps: int = 252,
    mean_reversion: float = 0.08,
    volatility: float = 1.0,
    jump_probability: float = 0.02,
    jump_scale: float = 2.5,
    random_state: int | None = None,
) -> pd.Series:
    """Generate an Ornstein-Uhlenbeck-like spread with jumps."""
    rng = np.random.default_rng(random_state)
    values = np.zeros(n_steps)
    for t in range(1, n_steps):
        shock = rng.normal(0, volatility)
        if rng.random() < jump_probability:
            shock += rng.normal(0, jump_scale * volatility)
        values[t] = values[t - 1] - mean_reversion * values[t - 1] + shock
    return pd.Series(values, index=pd.bdate_range(end=pd.Timestamp.today().normalize(), periods=n_steps), name="synthetic_spread")


def generate_synthetic_scenarios(
    n_scenarios: int = 50,
    n_steps: int = 252,
    base_mean_reversion: float = 0.08,
    base_volatility: float = 1.0,
    jump_probability: float = 0.02,
    jump_scale: float = 2.5,
    random_state: int = 42,
) -> dict[str, pd.Series]:
    """Generate multiple plausible spread scenarios for robustness tests."""
    rng = np.random.default_rng(random_state)
    scenarios = {}
    regimes = ["calm", "high_vol", "jump", "weak_mean_reversion", "stress"]
    for i in range(n_scenarios):
        regime = regimes[i % len(regimes)]
        mr = base_mean_reversion
        vol = base_volatility
        jp = jump_probability
        js = jump_scale
        if regime == "high_vol":
            vol *= 1.8
        elif regime == "jump":
            jp *= 3.0
            js *= 1.5
        elif regime == "weak_mean_reversion":
            mr *= 0.35
        elif regime == "stress":
            vol *= 2.2
            jp *= 4.0
            mr *= 0.25
        scenarios[f"scenario_{i+1:03d}_{regime}"] = generate_synthetic_spread(
            n_steps=n_steps,
            mean_reversion=mr,
            volatility=vol,
            jump_probability=jp,
            jump_scale=js,
            random_state=int(rng.integers(0, 1_000_000)),
        )
    return scenarios


def scenario_summary(scenarios: dict[str, pd.Series]) -> pd.DataFrame:
    rows = []
    for name, s in scenarios.items():
        rows.append({
            "scenario": name,
            "mean": float(s.mean()),
            "std": float(s.std()),
            "max_abs_spread": float(s.abs().max()),
            "autocorr_1": float(s.autocorr(lag=1)),
        })
    return pd.DataFrame(rows)
