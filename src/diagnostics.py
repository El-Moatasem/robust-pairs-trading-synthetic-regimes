from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.tsa.stattools import adfuller, coint
from statsmodels.stats.stattools import durbin_watson
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant


def adf_test(series: pd.Series, regression: str = "c") -> dict[str, Any]:
    """Run Augmented Dickey-Fuller test with safe handling for short series."""
    s = pd.Series(series).dropna()
    if len(s) < 30 or s.nunique() < 3:
        return {"adf_stat": np.nan, "pvalue": np.nan, "is_stationary_5pct": False}
    stat, pvalue, usedlag, nobs, crit, icbest = adfuller(s, regression=regression, autolag="AIC")
    return {
        "adf_stat": float(stat),
        "pvalue": float(pvalue),
        "used_lag": int(usedlag),
        "nobs": int(nobs),
        "critical_values": crit,
        "is_stationary_5pct": bool(pvalue < 0.05),
    }


def is_integrated_order_one(series: pd.Series) -> dict[str, Any]:
    """Check whether a price series is plausibly I(1)."""
    level = adf_test(series)
    diff = adf_test(pd.Series(series).diff().dropna())
    return {
        "level_pvalue": level.get("pvalue"),
        "diff_pvalue": diff.get("pvalue"),
        "plausibly_i1": bool((level.get("pvalue", 0) > 0.05) and (diff.get("pvalue", 1) < 0.05)),
    }


def engle_granger_coint(y: pd.Series, x: pd.Series) -> dict[str, Any]:
    """Run Engle-Granger cointegration test."""
    df = pd.concat([y, x], axis=1).dropna()
    if len(df) < 60:
        return {"coint_stat": np.nan, "pvalue": np.nan}
    stat, pvalue, crit = coint(df.iloc[:, 0], df.iloc[:, 1])
    return {"coint_stat": float(stat), "pvalue": float(pvalue), "critical_values": crit}


def estimate_hedge_ratio(y: pd.Series, x: pd.Series) -> float:
    """Estimate hedge ratio y = alpha + beta*x using OLS."""
    df = pd.concat([y, x], axis=1).dropna()
    if len(df) < 20:
        return np.nan
    model = OLS(df.iloc[:, 0], add_constant(df.iloc[:, 1])).fit()
    return float(model.params.iloc[1])


def residual_diagnostics(residuals: pd.Series) -> dict[str, Any]:
    """Check residual stationarity, autocorrelation, and approximate normality."""
    r = pd.Series(residuals).dropna()
    if len(r) < 30:
        return {"durbin_watson": np.nan, "adf_pvalue": np.nan, "normality_pvalue": np.nan}
    adf = adf_test(r)
    try:
        _, norm_p = stats.normaltest(r)
    except Exception:
        norm_p = np.nan
    return {
        "durbin_watson": float(durbin_watson(r)),
        "adf_pvalue": adf.get("pvalue"),
        "normality_pvalue": float(norm_p) if not np.isnan(norm_p) else np.nan,
    }


def phillips_ouliaris_proxy(y: pd.Series, x: pd.Series) -> dict[str, Any]:
    """Residual-based cointegration diagnostic used as a Phillips-Ouliaris-style proxy.

    Statsmodels does not expose a direct Phillips-Ouliaris test in all versions.
    For the capstone starter code, this function estimates the cointegrating
    regression and applies an ADF-style unit-root test to residuals. The report
    should describe it as a residual-based diagnostic unless a full PO test is
    added later.
    """
    beta = estimate_hedge_ratio(y, x)
    if np.isnan(beta):
        return {"beta": np.nan, "residual_adf_pvalue": np.nan, "cointegrated_proxy": False}
    resid = y.align(x, join="inner")[0] - beta * y.align(x, join="inner")[1]
    adf = adf_test(resid)
    return {"beta": beta, "residual_adf_pvalue": adf.get("pvalue"), "cointegrated_proxy": bool(adf.get("pvalue", 1) < 0.05)}
