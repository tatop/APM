# APM

Asset Performance Modeling toolkit for portfolio risk and single-factor analysis.

## Overview

APM provides utilities for analyzing portfolio performance using the single-index model (CAPM-style analysis). It calculates key metrics like alpha, beta, and R-squared for individual securities against benchmarks, and computes portfolio-level volatility components.

## Installation

```bash
pip install -e .
```

## Dependencies

- Python >= 3.13
- matplotlib >= 3.10.8
- numpy >= 2.4.0
- yfinance >= 1.0

## Modules

### singleFactor.py

Single-index regression for analyzing stock performance relative to a benchmark.

**Functions:**

- `single_index_regression(stock_ticker, benchmark_ticker, *, period, interval, plot)`:
  Downloads historical data and performs OLS regression to estimate:
  - **alpha**: Stock's excess return relative to benchmark
  - **beta**: Stock's sensitivity to market movements
  - **r_squared**: Model explanatory power
  - **observations**: Number of data points used

  Returns a dict with metrics and optional scatter plot.

**Example:**
```python
from src.singleFactor import single_index_regression

result = single_index_regression("NVDA", "^GSPC", period="1y")
print(result)
# {'alpha': 0.0012, 'beta': 1.45, 'r_squared': 0.68, 'observations': 252.0}
```

### volatility.py

Portfolio volatility decomposition into market and idiosyncratic components.

**Classes:**

- `Position`: Immutable dataclass representing a portfolio position
  - `notional`: Dollar value of position
  - **beta**: Market sensitivity
  - `idiosyncratic_volatility`: Asset-specific volatility (decimal)

**Functions:**

- `portfolio_volatility(positions, market_volatility)`:
  Decomposes portfolio risk into:
  - **portfolio_dollar_beta**: Weighted sum of position betas
  - **market_component**: Volatility from market exposure
  - **idio_component**: Volatility from asset-specific risk
  - **idio_component_var**: Idiosyncratic variance proportion
  - **total_volatility**: Combined portfolio volatility
  - **annual_market_risk**: Annualized market risk
  - **annual_idiosyncratic_risk**: Annualized idiosyncratic risk

**Example:**
```python
from src.volatility import Position, portfolio_volatility

positions = [
    Position(notional=10_000_000, beta=1.2, idiosyncratic_volatility=0.012),
    Position(notional=5_000_000, beta=0.7, idiosyncratic_volatility=0.005),
]
result = portfolio_volatility(positions, market_volatility=0.014)
```

## License

MIT
