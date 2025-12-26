from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf


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

    stock_returns = returns[stock_ticker].to_numpy()
    benchmark_returns = returns[benchmark_ticker].to_numpy()

    x = np.column_stack([np.ones_like(benchmark_returns), benchmark_returns])
    beta_vec, *_ = np.linalg.lstsq(x, stock_returns, rcond=None)
    alpha, beta = float(beta_vec[0]), float(beta_vec[1])

    fitted = alpha + beta * benchmark_returns
    resid = stock_returns - fitted
    ss_res = float(np.sum(resid**2))
    ss_tot = float(np.sum((stock_returns - stock_returns.mean()) ** 2))
    r_squared = 1.0 - ss_res / ss_tot if ss_tot else 0.0

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
        "observations": float(len(stock_returns)),
    }


def main() -> None:
    result = single_index_regression("NVDA", "^GSPC", period="1y")
    print(result)


if __name__ == "__main__":
    main()
