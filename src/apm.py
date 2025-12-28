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


def _render_table(
    headers: list[str],
    rows: list[list[str]],
    *,
    aligns: list[str] | None = None,
) -> str:
    if aligns is None:
        aligns = ["<"] * len(headers)
    if len(aligns) != len(headers):
        raise ValueError("aligns must match headers length.")
    if any(len(row) != len(headers) for row in rows):
        raise ValueError("All rows must have the same length as headers.")

    widths = [
        max(len(headers[i]), *(len(row[i]) for row in rows)) if rows else len(headers[i])
        for i in range(len(headers))
    ]

    def fmt_row(row: list[str]) -> str:
        parts = []
        for i, cell in enumerate(row):
            align = aligns[i]
            if align == ">":
                parts.append(cell.rjust(widths[i]))
            elif align == "^":
                parts.append(cell.center(widths[i]))
            else:
                parts.append(cell.ljust(widths[i]))
        return " | ".join(parts)

    header_line = fmt_row(headers)
    sep_line = "-+-".join("-" * w for w in widths)
    body = "\n".join(fmt_row(row) for row in rows) if rows else ""
    return "\n".join(line for line in (header_line, sep_line, body) if line)


def _print_table(
    title: str,
    headers: list[str],
    rows: list[list[str]],
    *,
    aligns: list[str] | None = None,
) -> None:
    try:
        from rich.console import Console
        from rich.table import Table
    except ImportError:
        print(title)
        print(_render_table(headers, rows, aligns=aligns))
        return

    console = Console()
    table = Table(title=title, show_header=True, header_style="bold")

    if aligns is None:
        aligns = ["<"] * len(headers)
    if len(aligns) != len(headers):
        raise ValueError("aligns must match headers length.")

    for header, align in zip(headers, aligns, strict=True):
        justify = "left"
        if align == ">":
            justify = "right"
        elif align == "^":
            justify = "center"
        table.add_column(header, justify=justify)

    for row in rows:
        table.add_row(*row)

    console.print(table)


def _fmt_float(value: float, *, decimals: int = 4) -> str:
    return f"{value:.{decimals}f}"


def _fmt_int(value: int) -> str:
    return f"{value:d}"


def _fmt_currency(value: float, *, decimals: int = 2) -> str:
    return f"${value:,.{decimals}f}"


def _load_returns(
    stock_ticker: str,
    benchmark_ticker: str,
    *,
    period: str,
    interval: str,
) -> tuple[np.ndarray, np.ndarray]:
    if stock_ticker == benchmark_ticker:
        prices = yf.download(
            stock_ticker,
            period=period,
            interval=interval,
            auto_adjust=True,
            progress=False,
        )
    else:
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
    if stock_ticker == benchmark_ticker:
        if returns.empty:
            raise ValueError("Not enough data to compute returns.")
        stock_returns = returns.to_numpy()
        benchmark_returns = stock_returns.copy()
    else:
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
    beta_vec = np.asarray(beta_vec).ravel()
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
    benchmark_ticker = "SPY"
    portfolio = {"NVDA": 10_000_000, "WMT": 5_000_000, "SPY": 10_000_000}
    regressions: dict[str, dict[str, float]] = {}

    for ticker in portfolio:
        regressions[ticker] = single_index_regression(
            ticker, benchmark_ticker, period="1y", plot=False
        )

    regression_rows: list[list[str]] = []
    for ticker in portfolio:
        reg = regressions[ticker]
        idio_key = f"{ticker}_daily_idiosyncratic_volatility"
        regression_rows.append(
            [
                ticker,
                _fmt_float(reg["alpha"]),
                _fmt_float(reg["beta"]),
                _fmt_float(reg["r_squared"]),
                _fmt_float(reg[idio_key]),
                _fmt_int(int(reg["observations"])),
            ]
        )

    _print_table(
        "Single index regressions (daily)",
        ["Ticker", "Alpha", "Beta", "R^2", "Idio vol", "Obs"],
        regression_rows,
        aligns=["<", ">", ">", ">", ">", ">"],
    )

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
    volatility_rows = [
        ["portfolio_dollar_beta", _fmt_currency(volatility["portfolio_dollar_beta"])],
        ["market_component", _fmt_currency(volatility["market_component"])],
        ["idio_component", _fmt_currency(volatility["idio_component"])],
        ["total_volatility", _fmt_currency(volatility["total_volatility"])],
        ["annual_market_risk", _fmt_currency(volatility["annual_market_risk"])],
        ["annual_idiosyncratic_risk", _fmt_currency(volatility["annual_idiosyncratic_risk"])],
    ]
    _print_table(
        "Portfolio volatility (dollar risk)",
        ["Metric", "Value"],
        volatility_rows,
        aligns=["<", ">"],
    )

    hedge = market_hedge(positions)
    exposures_rows = []
    for ticker, pos in zip(portfolio.keys(), positions, strict=True):
        exposures_rows.append(
            [
                ticker,
                _fmt_currency(pos.notional, decimals=0),
                _fmt_float(pos.beta),
                _fmt_currency(pos.notional * pos.beta, decimals=0),
            ]
        )
    _print_table(
        "Position exposures",
        ["Ticker", "Notional", "Beta", "Dollar beta"],
        exposures_rows,
        aligns=["<", ">", ">", ">"],
    )

    hedge_rows = [
        ["portfolio_dollar_beta", _fmt_currency(float(hedge["portfolio_dollar_beta"]))],
        ["market_hedge_nmv", _fmt_currency(float(hedge["market_hedge_nmv"]), decimals=0)],
        ["portfolio_beta", _fmt_float(float(hedge["portfolio_beta"]))],
        ["hedged_portfolio_beta", _fmt_float(float(hedge["hedged_portfolio_beta"]))],
    ]
    _print_table("Market hedge summary", ["Metric", "Value"], hedge_rows, aligns=["<", ">"])


if __name__ == "__main__":
    main()
