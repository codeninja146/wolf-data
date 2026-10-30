# wolf-data

Free market data for my Wolf's Morning Brief.

A GitHub Action runs `wolf_data.py` every weekday before the market opens. It saves prices, price history and option chains for my focus stocks into the `data/` folder:

- `data/summary.md`: the numbers as tables
- `data/metrics.json`: the same numbers as data
- `data/options.json`: option chains for the next two monthly expirations

Data comes from Yahoo Finance through yfinance, for personal use. Not investment advice.
