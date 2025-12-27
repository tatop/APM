from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

"""Market hedge utilities.

Example:
    positions = [
        Position(notional=10_000_000, beta=1.2),
        Position(notional=5_000_000, beta=0.7),
    ]
    result = market_hedge(positions)
    for key, value in result.items():
        print(f"{key}: {value}")
"""


@dataclass(frozen=True)
class Position:
    notional: float
    beta: float


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


def main() -> None:
    positions = [
        Position(notional=10_000_000, beta=1.2),
        Position(notional=5_000_000, beta=0.5),
        Position(notional=10_000_000, beta=1.0),
    ]
    result = market_hedge(positions)
    for key, value in result.items():
        if key.startswith("dollar_betas"):
            print(f"{key}: {value}")
        else:
            print(f"{key}: ${value:,.2f}".replace(",", "_"))


if __name__ == "__main__":
    main()
