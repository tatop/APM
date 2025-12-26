from __future__ import annotations

"""Portfolio volatility utilities.

Example:
    positions = [
        Position(notional=10_000_000, beta=1.2, idiosyncratic_volatility=0.012),
        Position(notional=5_000_000, beta=0.7, idiosyncratic_volatility=0.005),
        Position(notional=10_000_000, beta=1.0, idiosyncratic_volatility=0.00),
    ]
    result = portfolio_volatility(positions, market_volatility=0.014)
    for key, value in result.items():
        print(f"{key}: {value:,.4f}".replace(",", "_"))
"""

from dataclasses import dataclass
from typing import Iterable

import numpy as np


TRADING_DAYS = 252


@dataclass(frozen=True)
class Position:
    notional: float
    beta: float
    idiosyncratic_volatility: float


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
    idio_component_var = float((idio_component / total_volatility)**2)
    annual_idiosyncratic_risk = float(idio_component * np.sqrt(TRADING_DAYS))
    annual_market_risk = float(market_component * np.sqrt(TRADING_DAYS))


    return {
        "portfolio_dollar_beta": portfolio_dollar_beta,
        "market_component": market_component,
        "idio_component": idio_component,
        "idio_component_var": idio_component_var,
        "total_volatility": total_volatility,
        "annual_market_risk": annual_market_risk,
        "annual_idiosyncratic_risk": annual_idiosyncratic_risk,
    }


def main() -> None:
    positions = [
        Position(notional=10_000_000, beta=1.2, idiosyncratic_volatility=0.012),
        Position(notional=5_000_000, beta=0.7, idiosyncratic_volatility=0.005),
        Position(notional=10_000_000, beta=1.0, idiosyncratic_volatility=0.00),
    ]
    result = portfolio_volatility(positions, market_volatility=0.014)
    for key, value in result.items():
        print(f"{key}: {value:,.4f}".replace(",", "_"))


if __name__ == "__main__":
    main()
