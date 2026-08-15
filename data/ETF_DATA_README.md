# Strict real-data ETF replication

The final capstone includes an economically pre-specified ETF replication configuration in `config/config_m7_etf.yaml`.

The ETF universe is **real public market data only**. The experiment uses these pre-specified pair hypotheses before inspecting test-period profitability:

- SPY-IVV
- SPY-VOO
- IVV-VOO
- QQQ-QQQM
- GLD-IAU
- IWM-VTWO
- VTI-ITOT
- AGG-BND

The expected cache path is `data/raw/etf_prices.csv`. It is intentionally not populated with synthetic replacement data. Run with `--require-public-data`; if Yahoo Finance cannot be reached and the cache is absent, the run stops rather than fabricating results.

Commands:

```bash
rm -f data/raw/etf_prices.csv
python run_pipeline.py --config config/config_m7_etf.yaml --require-public-data
python scripts/verify_outputs.py --outputs outputs_m7_etf
python scripts/run_m7_analysis.py --config config/config_m7_etf.yaml --require-public-data
python scripts/summarize_etf_results.py --outputs outputs_m7_etf
```

If the CSV cache is already present, omit the `rm -f` line to reproduce the exact cached dataset without a network download.
