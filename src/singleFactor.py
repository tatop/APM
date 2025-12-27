from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf

"""Single index regression utilities.

Example:
    result = single_index_regression("NVDA", "^GSPC", period="1y")
    for key, value in result.items():
        print(f"{key}: {value:,.4f}".replace(",", "_"))
"""


def _load_returns(
    stock_ticker: str,
    benchmark_ticker: str,
    *,
    period: str,
    interval: str,
) -> tuple[np.ndarray, np.ndarray]:
    prices = yf.download(
        [stock_ticker, benchmark_ticker],
        period=period,
        interval=interval,
        auto_adjust=True,
        progress=False,
    )
    if prices.empty:
        raise ValueError("No data returned from yfinance.")

    close = prices["Close"] if "Close" in prices else prices["Adj Close"]
    returns = close.pct_change().dropna()
    returns = returns[[stock_ticker, benchmark_ticker]].dropna()
    if returns.empty:
        raise ValueError("Not enough data to compute returns.")

    stock_returns = returns[stock_ticker].to_numpy()
    benchmark_returns = returns[benchmark_ticker].to_numpy()
    return stock_returns, benchmark_returns


def _single_index_fit(
    stock_returns: np.ndarray,
    benchmark_returns: np.ndarray,
) -> tuple[float, float]:
    x = np.column_stack([np.ones_like(benchmark_returns), benchmark_returns])
    beta_vec, *_ = np.linalg.lstsq(x, stock_returns, rcond=None)
    alpha, beta = float(beta_vec[0]), float(beta_vec[1])
    return alpha, beta


def _plot_single_index(
    stock_returns: np.ndarray,
    benchmark_returns: np.ndarray,
    alpha: float,
    beta: float,
    *,
    stock_ticker: str,
    benchmark_ticker: str,
) -> None:
    plt.figure(figsize=(8, 6))
    plt.scatter(benchmark_returns, stock_returns, alpha=0.6, label="Daily returns")
    line_x = np.linspace(benchmark_returns.min(), benchmark_returns.max(), 200)
    line_y = alpha + beta * line_x
    plt.plot(line_x, line_y, color="crimson", label="Single index fit")
    plt.title(f"{stock_ticker} vs {benchmark_ticker} (daily returns)")
    plt.xlabel(f"{benchmark_ticker} return")
    plt.ylabel(f"{stock_ticker} return")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


def single_index_regression(
    stock_ticker: str,
    benchmark_ticker: str,
    *,
    period: str = "1y",
    interval: str = "1d",
    plot: bool = True,
) -> dict[str, float]:
    stock_returns, benchmark_returns = _load_returns(
        stock_ticker, benchmark_ticker, period=period, interval=interval
    )
    alpha, beta = _single_index_fit(stock_returns, benchmark_returns)

    fitted = alpha + beta * benchmark_returns
    resid = stock_returns - fitted
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((stock_returns - stock_returns.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot else 0.0
    observations = int(resid.size)
    denom = max(observations - 2, 1)
    idio_vol = float(np.sqrt(ss_res / denom))

    if plot:
        _plot_single_index(
            stock_returns,
            benchmark_returns,
            alpha,
            beta,
            stock_ticker=stock_ticker,
            benchmark_ticker=benchmark_ticker,
        )

    return {
        "alpha": alpha,
        "beta": beta,
        "r_squared": r_squared,
        f"{stock_ticker}_daily_idiosyncratic_volatility": idio_vol,
        "observations": float(observations),
    }


def main() -> None:
    stock_ticker = "NVDA"
    benchmark_ticker = "^GSPC"
    result = single_index_regression(stock_ticker, benchmark_ticker, period="1y")
    print("Single index regression:")
    for key, value in result.items():
        print(f"{key}: {value:,.4f}".replace(",", "_"))


if __name__ == "__main__":
    main()
