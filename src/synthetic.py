from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import pandas as pd
from statsmodels.api import OLS, add_constant

from .backtest import run_backtest, summarize_backtest
from .features import MODEL_FEATURE_COLUMNS, build_feature_frame
from .models import ModelBundle, _positive_probability


@dataclass(frozen=True)
class SyntheticCalibration:
    spread_mean: float
    spread_phi: float
    innovation_std: float
    jump_probability: float
    jump_std: float
    common_return_mean: float
    common_return_std: float

    def to_dict(self) -> dict:
        return asdict(self)


def calibrate_synthetic_process(training_features: pd.DataFrame) -> SyntheticCalibration:
    spread = training_features["spread"].dropna()
    lagged = spread.shift(1).dropna()
    current = spread.loc[lagged.index]
    fit = OLS(current, add_constant(lagged)).fit()
    intercept = float(fit.params.iloc[0])
    phi = float(np.clip(fit.params.iloc[1], 0.05, 0.995))
    mean = float(intercept / (1.0 - phi))
    residual = pd.Series(fit.resid, index=current.index)
    innovation_std = float(max(residual.std(ddof=0), 1e-5))
    jump_threshold = float(residual.abs().quantile(0.99))
    jumps = residual[residual.abs() >= jump_threshold]
    jump_probability = float(max(len(jumps) / max(len(residual), 1), 0.002))
    jump_std = float(max(jumps.std(ddof=0) if len(jumps) > 2 else 3.0 * innovation_std, 2.0 * innovation_std))
    common_returns = np.log(training_features["asset_b_price"]).diff().dropna()
    return SyntheticCalibration(
        spread_mean=mean,
        spread_phi=phi,
        innovation_std=innovation_std,
        jump_probability=jump_probability,
        jump_std=jump_std,
        common_return_mean=float(common_returns.mean()),
        common_return_std=float(max(common_returns.std(ddof=0), 1e-5)),
    )


def _simulate_pair(
    length: int,
    calibration: SyntheticCalibration,
    regime: dict,
    alpha: float,
    beta: float,
    start_asset_b: float,
    start_date: pd.Timestamp,
    rng: np.random.Generator,
    asset_a: str,
    asset_b: str,
) -> pd.DataFrame:
    vol_mult = float(regime["volatility_multiplier"])
    mr_mult = float(regime["mean_reversion_multiplier"])
    jump_prob_mult = float(regime["jump_probability_multiplier"])
    jump_size_mult = float(regime["jump_size_multiplier"])
    mean_shift_std = float(regime["long_run_mean_shift_std"])

    base_phi = calibration.spread_phi
    base_kappa = max(1.0 - base_phi, 1e-4)
    scenario_kappa = np.clip(base_kappa * mr_mult, 1e-4, 0.95)
    scenario_phi = 1.0 - scenario_kappa
    shifted_mean = calibration.spread_mean + rng.normal(0.0, mean_shift_std * calibration.innovation_std)
    innovation_std = calibration.innovation_std * vol_mult
    jump_probability = min(calibration.jump_probability * jump_prob_mult, 0.25)
    jump_std = calibration.jump_std * jump_size_mult

    spread = np.zeros(length, dtype=float)
    spread[0] = shifted_mean + rng.normal(0.0, innovation_std)
    for i in range(1, length):
        innovation = rng.normal(0.0, innovation_std)
        if rng.random() < jump_probability:
            innovation += rng.normal(0.0, jump_std)
        spread[i] = shifted_mean + scenario_phi * (spread[i - 1] - shifted_mean) + innovation

    common_returns = rng.normal(
        calibration.common_return_mean,
        calibration.common_return_std * max(1.0, np.sqrt(vol_mult)),
        length,
    )
    log_b = np.log(start_asset_b) + np.cumsum(common_returns)
    log_a = alpha + beta * log_b + spread
    dates = pd.bdate_range(start=start_date, periods=length)
    return pd.DataFrame({asset_a: np.exp(log_a), asset_b: np.exp(log_b)}, index=dates)


def evaluate_synthetic_regimes(
    training_features: pd.DataFrame,
    cfg: dict,
    asset_a: str,
    asset_b: str,
    alpha: float,
    beta: float,
    selected_bundle: ModelBundle,
) -> tuple[pd.DataFrame, pd.DataFrame, SyntheticCalibration]:
    calibration = calibrate_synthetic_process(training_features)
    scfg = cfg["synthetic"]
    rng = np.random.default_rng(int(scfg.get("seed", 42)))
    regimes = list(scfg["regimes"].items())
    scenario_rows: list[dict[str, Any]] = []
    aggregate_frames: list[pd.DataFrame] = []
    start_date = pd.Timestamp("2030-01-02")

    for scenario_id in range(int(scfg["n_scenarios"])):
        regime_name, regime = regimes[scenario_id % len(regimes)]
        prices = _simulate_pair(
            length=int(scfg["scenario_length"]),
            calibration=calibration,
            regime=regime,
            alpha=alpha,
            beta=beta,
            start_asset_b=float(training_features["asset_b_price"].iloc[-1]),
            start_date=start_date + pd.offsets.BDay(scenario_id * (int(scfg["scenario_length"]) + 5)),
            rng=rng,
            asset_a=asset_a,
            asset_b=asset_b,
        )
        scenario_features = build_feature_frame(prices, asset_a, asset_b, cfg, alpha, beta)
        if len(scenario_features) < 30:
            continue
        probability = _positive_probability(selected_bundle.model, scenario_features[MODEL_FEATURE_COLUMNS])
        accept = pd.Series(probability >= selected_bundle.threshold, index=scenario_features.index)
        baseline, baseline_trades = run_backtest(
            scenario_features, cfg, strategy_name="synthetic_baseline", accept_filter=None
        )
        filtered, filtered_trades = run_backtest(
            scenario_features, cfg, strategy_name="synthetic_ml_filtered", accept_filter=accept
        )
        baseline_summary = summarize_backtest(
            baseline, baseline_trades, "synthetic_baseline", cfg, seed=scenario_id + 1000, compute_ci=False
        )
        filtered_summary = summarize_backtest(
            filtered, filtered_trades, "synthetic_ml_filtered", cfg, seed=scenario_id + 2000, compute_ci=False
        )
        spread = scenario_features["spread"]
        scenario_rows.append(
            {
                "scenario_id": scenario_id + 1,
                "regime": regime_name,
                "spread_std": float(spread.std(ddof=0)),
                "spread_max_abs_deviation": float((spread - spread.mean()).abs().max()),
                "spread_autocorrelation_1": float(spread.autocorr(1)),
                "baseline_cumulative_net_pnl": baseline_summary["cumulative_net_pnl"],
                "baseline_sharpe": baseline_summary["sharpe"],
                "baseline_max_drawdown": baseline_summary["max_drawdown"],
                "ml_cumulative_net_pnl": filtered_summary["cumulative_net_pnl"],
                "ml_sharpe": filtered_summary["sharpe"],
                "ml_max_drawdown": filtered_summary["max_drawdown"],
                "ml_minus_baseline_pnl": filtered_summary["cumulative_net_pnl"]
                - baseline_summary["cumulative_net_pnl"],
            }
        )
        if scenario_id < 3:
            copy = scenario_features[["spread", "signal_zscore"]].copy()
            copy["scenario_id"] = scenario_id + 1
            copy["regime"] = regime_name
            aggregate_frames.append(copy.reset_index(names="date"))

    scenario_table = pd.DataFrame(scenario_rows)
    sample_paths = pd.concat(aggregate_frames, ignore_index=True) if aggregate_frames else pd.DataFrame()
    return scenario_table, sample_paths, calibration


def summarize_regimes(scenario_table: pd.DataFrame) -> pd.DataFrame:
    if scenario_table.empty:
        return pd.DataFrame()
    return (
        scenario_table.groupby("regime", as_index=False)
        .agg(
            scenarios=("scenario_id", "count"),
            mean_spread_std=("spread_std", "mean"),
            mean_baseline_pnl=("baseline_cumulative_net_pnl", "mean"),
            mean_ml_pnl=("ml_cumulative_net_pnl", "mean"),
            median_baseline_sharpe=("baseline_sharpe", "median"),
            median_ml_sharpe=("ml_sharpe", "median"),
            mean_ml_minus_baseline_pnl=("ml_minus_baseline_pnl", "mean"),
            ml_outperforms_fraction=("ml_minus_baseline_pnl", lambda s: float((s > 0).mean())),
        )
        .sort_values("regime")
    )
