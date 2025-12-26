import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.singleFactor import daily_idiosyncratic_volatility, single_index_regression


def _prices_from_returns(returns: list[float], start: float = 100.0) -> np.ndarray:
    prices = [start]
    for value in returns:
        prices.append(prices[-1] * (1.0 + value))
    return np.array(prices, dtype=float)


class TestSingleFactor(unittest.TestCase):
    def test_daily_idiosyncratic_volatility_perfect_fit(self) -> None:
        benchmark_returns = [0.01, -0.02, 0.03, 0.00]
        stock_returns = [0.001 + 2.0 * r for r in benchmark_returns]

        prices = pd.DataFrame(
            {
                ("Close", "STOCK"): _prices_from_returns(stock_returns),
                ("Close", "BENCH"): _prices_from_returns(benchmark_returns),
            },
            index=pd.date_range("2024-01-01", periods=5, freq="D"),
        )
        prices.columns = pd.MultiIndex.from_tuples(prices.columns)

        with patch("src.singleFactor.yf.download", return_value=prices):
            result = daily_idiosyncratic_volatility("STOCK", "BENCH", period="5d")

        self.assertAlmostEqual(result["daily_idiosyncratic_volatility"], 0.0, places=10)
        self.assertAlmostEqual(result["alpha"], 0.001, places=10)
        self.assertAlmostEqual(result["beta"], 2.0, places=10)
        self.assertEqual(result["observations"], 4.0)

    def test_single_index_regression_r_squared_perfect_fit(self) -> None:
        benchmark_returns = [0.01, -0.02, 0.03, 0.00]
        stock_returns = [0.002 + 1.5 * r for r in benchmark_returns]

        prices = pd.DataFrame(
            {
                ("Close", "AAA"): _prices_from_returns(stock_returns),
                ("Close", "SPX"): _prices_from_returns(benchmark_returns),
            },
            index=pd.date_range("2024-02-01", periods=5, freq="D"),
        )
        prices.columns = pd.MultiIndex.from_tuples(prices.columns)

        with patch("src.singleFactor.yf.download", return_value=prices):
            result = single_index_regression("AAA", "SPX", period="5d", plot=False)

        self.assertAlmostEqual(result["alpha"], 0.002, places=10)
        self.assertAlmostEqual(result["beta"], 1.5, places=10)
        self.assertAlmostEqual(result["r_squared"], 1.0, places=10)
        self.assertEqual(result["observations"], 4.0)


if __name__ == "__main__":
    unittest.main()
