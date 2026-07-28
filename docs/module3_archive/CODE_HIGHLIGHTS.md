# Important Code Sections

These are the main code sections to highlight in the project proposal report and future submissions.

| File / module | Purpose |
|---|---|
| `run_pipeline.py` | End-to-end workflow that loads data, screens pairs, builds features, trains models, backtests strategies, generates synthetic regimes, and exports outputs. |
| `single_file_demo.py` | Minimal self-contained demo that can run without internet access or project imports. |
| `src/data.py` | Loads public equity/ETF data with an offline synthetic-data fallback. |
| `src/pairs.py` | Screens candidate pairs using return correlation, Engle-Granger cointegration, ADF tests, and hedge-ratio estimation. |
| `src/features.py` | Builds spread, z-score, rolling volatility, rolling correlation, drawdown, half-life, persistence, and stress-proxy features. |
| `src/labels.py` | Creates accept/reject trade labels based on convergence within a horizon after transaction costs and stop-loss constraints. |
| `src/models.py` | Trains logistic regression, random forest, and optional XGBoost or fallback gradient-boosting trade filters. |
| `src/backtest.py` | Compares baseline z-score and ML-filtered trading variants with costs, slippage, stops, and max holding period. |
| `src/synthetic.py` | Generates synthetic spread regimes for calm, high-volatility, jump, weak-mean-reversion, and stress scenarios. |
| `src/visualization.py` | Generates report-ready figures with titles, axes, and labels. |
| `outputs/` | Stores preliminary tables, figures, and initial findings used in the report. |
