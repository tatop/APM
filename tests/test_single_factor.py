import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.apm import single_index_regression


def _prices_from_returns(returns: list[float], start: float = 100.0) -> np.ndarray:
    """Convert a list of returns to prices starting from a given value."""
    prices = [start]
    for value in returns:
        prices.append(prices[-1] * (1.0 + value))
    return np.array(prices, dtype=float)


def _create_mock_prices(
    stock_ticker: str,
    benchmark_ticker: str,
    stock_returns: list[float],
    benchmark_returns: list[float],
    start_date: str = "2024-01-01",
) -> pd.DataFrame:
    """Create a mock DataFrame mimicking yfinance download output."""
    periods = len(stock_returns) + 1
    prices = pd.DataFrame(
        {
            ("Close", stock_ticker): _prices_from_returns(stock_returns),
            ("Close", benchmark_ticker): _prices_from_returns(benchmark_returns),
        },
        index=pd.date_range(start_date, periods=periods, freq="D"),
    )
    prices.columns = pd.MultiIndex.from_tuples(prices.columns)
    return prices


class TestSingleIndexRegression(unittest.TestCase):
    def test_perfect_fit_zero_idiosyncratic_volatility(self) -> None:
        """When stock returns are a perfect linear function of benchmark, idio vol should be zero."""
        benchmark_returns = [0.01, -0.02, 0.03, 0.00]
        alpha, beta = 0.001, 2.0
        stock_returns = [alpha + beta * r for r in benchmark_returns]

        prices = _create_mock_prices("STOCK", "BENCH", stock_returns, benchmark_returns)

        with patch("src.apm.yf.download", return_value=prices):
            result = single_index_regression("STOCK", "BENCH", period="5d", plot=False)

        self.assertAlmostEqual(result["alpha"], alpha, places=10)
        self.assertAlmostEqual(result["beta"], beta, places=10)
        self.assertAlmostEqual(result["r_squared"], 1.0, places=10)
        self.assertAlmostEqual(
            result["STOCK_daily_idiosyncratic_volatility"], 0.0, places=10
        )
        self.assertEqual(result["observations"], 4.0)

    def test_perfect_fit_different_parameters(self) -> None:
        """Test with different alpha and beta values."""
        benchmark_returns = [0.01, -0.02, 0.03, 0.00, 0.015]
        alpha, beta = 0.002, 1.5
        stock_returns = [alpha + beta * r for r in benchmark_returns]

        prices = _create_mock_prices(
            "AAA", "SPX", stock_returns, benchmark_returns, "2024-02-01"
        )

        with patch("src.apm.yf.download", return_value=prices):
            result = single_index_regression("AAA", "SPX", period="5d", plot=False)

        self.assertAlmostEqual(result["alpha"], alpha, places=10)
        self.assertAlmostEqual(result["beta"], beta, places=10)
        self.assertAlmostEqual(result["r_squared"], 1.0, places=10)
        self.assertAlmostEqual(
            result["AAA_daily_idiosyncratic_volatility"], 0.0, places=10
        )
        self.assertEqual(result["observations"], 5.0)

    def test_imperfect_fit_positive_idiosyncratic_volatility(self) -> None:
        """When there is noise in returns, idio vol should be positive and r_squared < 1."""
        benchmark_returns = [0.01, -0.02, 0.03, 0.00]
        noise = [0.005, -0.003, 0.002, -0.004]
        alpha, beta = 0.001, 1.0
        stock_returns = [alpha + beta * r + n for r, n in zip(benchmark_returns, noise)]

        prices = _create_mock_prices("NOISY", "BENCH", stock_returns, benchmark_returns)

        with patch("src.apm.yf.download", return_value=prices):
            result = single_index_regression("NOISY", "BENCH", period="5d", plot=False)

        self.assertGreater(result["NOISY_daily_idiosyncratic_volatility"], 0.0)
        self.assertLess(result["r_squared"], 1.0)
        self.assertEqual(result["observations"], 4.0)

    def test_beta_zero_no_correlation(self) -> None:
        """When stock returns are constant, beta should be close to zero."""
        benchmark_returns = [0.01, -0.02, 0.03, -0.01]
        constant_return = 0.005
        stock_returns = [constant_return] * len(benchmark_returns)

        prices = _create_mock_prices("CONST", "BENCH", stock_returns, benchmark_returns)

        with patch("src.apm.yf.download", return_value=prices):
            result = single_index_regression("CONST", "BENCH", period="5d", plot=False)

        self.assertAlmostEqual(result["beta"], 0.0, places=10)
        self.assertAlmostEqual(result["alpha"], constant_return, places=10)
        self.assertEqual(result["observations"], 4.0)

    def test_negative_beta(self) -> None:
        """Test that negative beta is correctly computed for inverse relationship."""
        benchmark_returns = [0.01, -0.02, 0.03, 0.00]
        alpha, beta = 0.0, -1.5
        stock_returns = [alpha + beta * r for r in benchmark_returns]

        prices = _create_mock_prices("INV", "BENCH", stock_returns, benchmark_returns)

        with patch("src.apm.yf.download", return_value=prices):
            result = single_index_regression("INV", "BENCH", period="5d", plot=False)

        self.assertAlmostEqual(result["beta"], beta, places=10)
        self.assertAlmostEqual(result["alpha"], alpha, places=10)
        self.assertAlmostEqual(result["r_squared"], 1.0, places=10)

    def test_empty_data_raises_error(self) -> None:
        """Test that empty data raises ValueError."""
        empty_prices = pd.DataFrame()

        with patch("src.apm.yf.download", return_value=empty_prices):
            with self.assertRaises(ValueError) as context:
                single_index_regression("EMPTY", "BENCH", period="5d", plot=False)

        self.assertIn("No data returned", str(context.exception))

    def test_result_keys_include_ticker_name(self) -> None:
        """Test that the idiosyncratic volatility key includes the stock ticker."""
        benchmark_returns = [0.01, -0.02, 0.03]
        stock_returns = [0.02, -0.01, 0.04]

        prices = _create_mock_prices(
            "MYTICKER", "SPY", stock_returns, benchmark_returns
        )

        with patch("src.apm.yf.download", return_value=prices):
            result = single_index_regression("MYTICKER", "SPY", period="5d", plot=False)

        expected_keys = {
            "alpha",
            "beta",
            "r_squared",
            "MYTICKER_daily_idiosyncratic_volatility",
            "observations",
        }
        self.assertEqual(set(result.keys()), expected_keys)


if __name__ == "__main__":
    unittest.main()
