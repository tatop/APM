import unittest

from src.marketHedge import Position, market_hedge


class TestMarketHedge(unittest.TestCase):
    def test_market_hedge_values(self) -> None:
        positions = [
            Position(notional=10_000_000, beta=1.2),
            Position(notional=5_000_000, beta=0.7),
        ]

        result = market_hedge(positions)

        expected_dollar_betas = [12_000_000.0, 3_500_000.0]
        expected_portfolio_beta = 15_500_000.0
        expected_hedge_nmv = -15_500_000.0
        expected_beta = 15_500_000.0 / 15_000_000.0

        self.assertEqual(result["dollar_betas"], expected_dollar_betas)
        self.assertAlmostEqual(
            result["portfolio_dollar_beta"], expected_portfolio_beta, places=6
        )
        self.assertAlmostEqual(
            result["market_hedge_nmv"], expected_hedge_nmv, places=6
        )
        self.assertAlmostEqual(result["portfolio_beta"], expected_beta, places=10)
        self.assertAlmostEqual(result["hedged_portfolio_beta"], 0.0, places=10)

    def test_empty_positions_raises(self) -> None:
        with self.assertRaises(ValueError) as context:
            market_hedge([])

        self.assertIn("At least one position", str(context.exception))

    def test_zero_total_nmv_raises(self) -> None:
        positions = [
            Position(notional=10_000.0, beta=1.0),
            Position(notional=-10_000.0, beta=1.0),
        ]

        with self.assertRaises(ValueError) as context:
            market_hedge(positions)

        self.assertIn("Total portfolio NMV must be non-zero", str(context.exception))


if __name__ == "__main__":
    unittest.main()
