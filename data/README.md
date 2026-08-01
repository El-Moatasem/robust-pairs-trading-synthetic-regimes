# Data directory

The submitted M6 ZIP does not need to contain private or externally downloaded market data.

For strict public-data mode, either:

1. allow `yfinance` to download the configured tickers, or
2. place a clean adjusted-close CSV at `data/raw/public_prices.csv`.

The first column should be a parseable date index and the remaining columns should be ticker price series. The pipeline records whether the final run was synthetic or public in `outputs*/tables/data_summary.json`.
