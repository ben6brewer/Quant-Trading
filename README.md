# Quantitative Trading Project

This is a structured template for a quantitative systematic trading project using Python, Alpaca API, and fundamental/momentum strategies.

## Setup

- Create a `.env` file with your Alpaca API keys.
- Install requirements: `pip install -r requirements.txt`
- Run `run.py` to backtest or trade live.

## Project Structure

- `config/` - configuration and secrets
- `strategies/` - trading strategies implementations
- `backtest/` - backtesting engine and metrics
- `execution/` - live trading interface with Alpaca
- `utils/` - data fetching and helper utilities
- `notebooks/` - Jupyter notebooks for exploration and prototyping
- `tests/` - unit tests

## Notes

Do NOT commit your `.env` file!
