from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from typing import Iterable

try:
    import distutils  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover - only needed for Python 3.13+
    import sys

    import setuptools._distutils as _distutils

    sys.modules.setdefault("distutils", _distutils)

import numpy as np
import pandas as pd
import polars as pl
import statsmodels.api as sm
import yfinance as yf
from pandas_datareader import data as pdr


DEFAULT_START = "2018-01-01"
DEFAULT_END = "2019-01-01"
DEFAULT_INTERVAL = "1d"


@dataclass(frozen=True)
class FactorExposure:
    alpha: float
    market_beta: float
    smb: float
    hml: float
    r_squared: float
    observations: int


def _normalize_weights(holdings: dict[str, float]) -> dict[str, float]:
    total = float(sum(holdings.values()))
    if total <= 0:
        raise ValueError("Total notional must be positive.")
    return {ticker: notional / total for ticker, notional in holdings.items()}


def _download_prices(
    tickers: Iterable[str],
    interval: str,
    start: str | date | None,
    end: str | date | None,
) -> pl.DataFrame:
    prices = yf.download(
        list(tickers),
        interval=interval,
        start=start,
        end=end,
        progress=False,
    )["Close"]
    if prices.empty:
        raise ValueError("No data returned from yfinance.")

    close = prices.dropna(how="all")
    if close.empty:
        raise ValueError("No valid closing prices returned from yfinance.")

    return pl.from_pandas(close.reset_index())


def _compute_returns(prices: pl.DataFrame, tickers: list[str]) -> pl.DataFrame:
    prices = prices.sort("Date")
    exprs = [pl.col("Date")]
    for ticker in tickers:
        exprs.append(pl.col(ticker).pct_change().alias(ticker))
    returns = prices.select(exprs).drop_nulls()
    returns = _normalize_date_column(returns)
    if returns.is_empty():
        raise ValueError("Not enough data to compute returns.")
    return returns


def _load_fama_french_factors(
    *,
    start: str | date | None,
    end: str | date | None,
) -> pl.DataFrame:
    factor_data = pdr.DataReader(
        "F-F_Research_Data_Factors_daily",
        "famafrench",
        start=start,
        end=end,
    )[0]
    factor_data = factor_data.reset_index()
    if "index" in factor_data.columns and "Date" not in factor_data.columns:
        factor_data = factor_data.rename(columns={"index": "Date"})
    factor_data["Date"] = pd.to_datetime(
        factor_data["Date"],
        errors="coerce",
        format="%Y%m%d",
    )
    factors = pl.from_pandas(factor_data)
    return _normalize_date_column(factors)


def _normalize_date_column(frame: pl.DataFrame) -> pl.DataFrame:
    date_str = pl.col("Date").cast(pl.Utf8, strict=False)
    parsed_yyyymmdd = date_str.str.strptime(pl.Date, "%Y%m%d", strict=False)
    normalized = frame.with_columns(
        pl.when(date_str.str.len_chars() == 8)
        .then(parsed_yyyymmdd)
        .otherwise(pl.col("Date").cast(pl.Date, strict=False))
        .alias("Date")
    )
    return normalized.drop_nulls(["Date"])


def _format_date_for_join(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.with_columns(
        pl.col("Date").cast(pl.Date, strict=False).dt.strftime("%Y-%m-%d").alias("_join_date")
    )


def _prepare_regression_frame(
    returns: pl.DataFrame,
    factors: pl.DataFrame,
    weights: dict[str, float],
) -> pl.DataFrame:
    weighted = None
    for ticker, weight in weights.items():
        term = pl.col(ticker) * weight
        weighted = term if weighted is None else weighted + term

    merged = returns.join(factors, on="Date", how="inner")
    merged = merged.with_columns(
        weighted.alias("portfolio_return"),
        (pl.col("Mkt-RF") / 100.0).alias("Mkt-RF"),
        (pl.col("SMB") / 100.0).alias("SMB"),
        (pl.col("HML") / 100.0).alias("HML"),
        (pl.col("RF") / 100.0).alias("RF"),
    )
    merged = merged.with_columns(
        (pl.col("portfolio_return") - pl.col("RF")).alias("excess_return")
    )
    if merged.is_empty():
        returns_join = _format_date_for_join(returns)
        factors_join = _format_date_for_join(factors)
        merged = returns_join.join(factors_join, on="_join_date", how="inner")
        merged = merged.with_columns(
            weighted.alias("portfolio_return"),
            (pl.col("Mkt-RF") / 100.0).alias("Mkt-RF"),
            (pl.col("SMB") / 100.0).alias("SMB"),
            (pl.col("HML") / 100.0).alias("HML"),
            (pl.col("RF") / 100.0).alias("RF"),
            pl.col("_join_date").str.strptime(pl.Date, "%Y-%m-%d", strict=False).alias("Date"),
        )
        merged = merged.with_columns(
            (pl.col("portfolio_return") - pl.col("RF")).alias("excess_return")
        )
    if merged.is_empty():
        returns_range = returns.select(
            pl.min("Date").alias("min"), pl.max("Date").alias("max")
        ).to_dicts()
        factors_range = factors.select(
            pl.min("Date").alias("min"), pl.max("Date").alias("max")
        ).to_dicts()
        raise ValueError(
            "No overlapping data between returns and factors. "
            f"returns_range={returns_range} factors_range={factors_range}"
        )
    return merged


def regress_portfolio_factor_exposure(
    holdings: dict[str, float],
    *,
    interval: str = DEFAULT_INTERVAL,
    start: str | date | None = DEFAULT_START,
    end: str | date | None = DEFAULT_END,
) -> FactorExposure:
    weights = _normalize_weights(holdings)
    tickers = list(weights)

    prices = _download_prices(tickers, interval=interval, start=start, end=end)
    returns = _compute_returns(prices, tickers)
    factors = _load_fama_french_factors(start=start, end=end)
    frame = _prepare_regression_frame(returns, factors, weights)

    y = frame.select("excess_return").to_numpy().ravel()
    x = frame.select(["Mkt-RF", "SMB", "HML"]).to_numpy()
    x = sm.add_constant(x)
    model = sm.OLS(y, x).fit()

    alpha, market, smb, hml = (float(value) for value in model.params)
    return FactorExposure(
        alpha=alpha,
        market_beta=market,
        smb=smb,
        hml=hml,
        r_squared=float(model.rsquared),
        observations=int(model.nobs),
    )


def main() -> None:
    holdings = {
        "VFMV": 2_500_000,
    }
    exposure = regress_portfolio_factor_exposure(
        holdings,
        interval=DEFAULT_INTERVAL,
        start=DEFAULT_START,
        end=DEFAULT_END,
    )
    print("Fama-French 3-factor exposure (daily):")
    print(f"alpha: {exposure.alpha:.6f}")
    print(f"market_beta: {exposure.market_beta:.4f}")
    print(f"smb: {exposure.smb:.4f}")
    print(f"hml: {exposure.hml:.4f}")
    print(f"r_squared: {exposure.r_squared:.4f}")
    print(f"observations: {exposure.observations}")


if __name__ == "__main__":
    main()
