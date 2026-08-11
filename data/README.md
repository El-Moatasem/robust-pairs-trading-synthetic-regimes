# Data directory

The final submission includes the public verification cache used for the reported 10-asset experiment:

```text
data/raw/public_prices.csv
```

It contains adjusted-close daily prices for AAPL, BAC, CVX, JPM, KO, MSFT, PEP, QQQ, SPY, and XOM from 2018-01-02 through 2026-06-29. The pipeline validates that requested tickers are present and reorders cached columns to the configured ticker list before analysis.

Strict public mode can also download a fresh dataset with `yfinance` if the cache is removed and internet access is available. It does not silently fall back to synthetic data.

The optional ETF extension uses:

```text
data/raw/etf_prices.csv
```

That ETF cache is not included because the ETF extension was not used to produce the reported final conclusions. Running `config/config_m7_etf.yaml` with internet access will create it automatically.
