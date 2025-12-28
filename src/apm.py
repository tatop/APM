from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import yfinance as yf

"""Asset pricing model utilities.

Example:
    portfolio = {"NVDA": 10_000_000, "WMT": 5_000_000, "SPY": 10_000_000}
    regressions = {
        ticker: single_index_regression(ticker, "^GSPC", period="1y", plot=False)
        for ticker in portfolio
    }
    positions = [
        Position(
            notional=notional,
            beta=regressions[ticker]["beta"],
            idiosyncratic_volatility=regressions[ticker][
                f"{ticker}_daily_idiosyncratic_volatility"
            ],
        )
        for ticker, notional in portfolio.items()
    ]
    vol = portfolio_volatility(positions, market_volatility=0.014)
    hedge = market_hedge(positions)
"""


TRADING_DAYS = 252


@dataclass(frozen=True)
class Position:
    notional: float
    beta: float
    idiosyncratic_volatility: float = 0.0


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


def market_hedge(positions: Iterable[Position]) -> dict[str, float | list[float]]:
    positions_list = list(positions)
    if not positions_list:
        raise ValueError("At least one position is required.")

    notionals = np.array([pos.notional for pos in positions_list], dtype=float)
    betas = np.array([pos.beta for pos in positions_list], dtype=float)

    total_nmv = float(np.sum(notionals))
    if total_nmv == 0.0:
        raise ValueError("Total portfolio NMV must be non-zero.")

    dollar_betas = notionals * betas
    portfolio_dollar_beta = float(np.sum(dollar_betas))
    market_hedge_nmv = -portfolio_dollar_beta
    portfolio_beta = portfolio_dollar_beta / total_nmv
    hedged_dollar_beta = portfolio_dollar_beta + market_hedge_nmv
    hedged_total_nmv = total_nmv + market_hedge_nmv
    hedged_portfolio_beta = (
        hedged_dollar_beta / hedged_total_nmv if hedged_total_nmv != 0.0 else 0.0
    )

    return {
        "dollar_betas": dollar_betas.tolist(),
        "portfolio_dollar_beta": portfolio_dollar_beta,
        "market_hedge_nmv": market_hedge_nmv,
        "portfolio_beta": portfolio_beta,
        "hedged_portfolio_beta": hedged_portfolio_beta,
    }


def portfolio_volatility(
    positions: Iterable[Position],
    market_volatility: float,
) -> dict[str, float]:
    positions_list = list(positions)
    if not positions_list:
        raise ValueError("At least one position is required.")

    notionals = np.array([pos.notional for pos in positions_list], dtype=float)
    betas = np.array([pos.beta for pos in positions_list], dtype=float)
    idio_vols = np.array(
        [pos.idiosyncratic_volatility for pos in positions_list], dtype=float
    )

    dollar_betas = notionals * betas
    portfolio_dollar_beta = float(np.sum(dollar_betas))

    market_component = portfolio_dollar_beta * market_volatility

    dollar_idio_vols = notionals * idio_vols
    idio_component = float(np.sqrt(np.sum(dollar_idio_vols**2)))

    total_volatility = float(np.sqrt(market_component**2 + idio_component**2))
    annual_idiosyncratic_risk = float(idio_component * np.sqrt(TRADING_DAYS))
    annual_market_risk = float(market_component * np.sqrt(TRADING_DAYS))

    return {
        "portfolio_dollar_beta": portfolio_dollar_beta,
        "market_component": market_component,
        "idio_component": idio_component,
        "total_volatility": total_volatility,
        "annual_market_risk": annual_market_risk,
        "annual_idiosyncratic_risk": annual_idiosyncratic_risk,
    }


def main() -> None:
    benchmark_ticker = "^GSPC"
    portfolio = {"NVDA": 10_000_000, "WMT": 5_000_000, "SPY": 10_000_000}
    regressions: dict[str, dict[str, float]] = {}

    for ticker in portfolio:
        regressions[ticker] = single_index_regression(
            ticker, benchmark_ticker, period="1y", plot=False
        )

    print("Single index regressions:")
    for ticker, regression in regressions.items():
        print(f"\n{ticker}:")
        for key, value in regression.items():
            print(f"{key}: {value:,.4f}".replace(",", "_"))

    positions = [
        Position(
            notional=notional,
            beta=regressions[ticker]["beta"],
            idiosyncratic_volatility=regressions[ticker][
                f"{ticker}_daily_idiosyncratic_volatility"
            ],
        )
        for ticker, notional in portfolio.items()
    ]

    volatility = portfolio_volatility(positions, market_volatility=0.014)
    print("\nPortfolio volatility:")
    for key, value in volatility.items():
        if key.endswith("var"):
            print(f"{key}: {value:,.4f}".replace(",", "_"))
        else:
            print(f"{key}: ${value:,.2f}".replace(",", "_"))

    hedge = market_hedge(positions)
    print("\nMarket hedge:")
    for key, value in hedge.items():
        if key.startswith("dollar_betas"):
            print(f"{key}: {value}")
        else:
            print(f"{key}: ${value:,.2f}".replace(",", "_"))


if __name__ == "__main__":
    main()
