import numpy as np
import polars as pl
import statsmodels.api as sm
import yfinance as yf

# 1. Define your Tickers and Weights
# Replace these with your specific ETF tickers if different
tickers = {
    "MinVol": "ACWV",  # iShares MSCI Global Min Vol
    "Momentum": "MTUM",  # iShares MSCI USA Momentum (or 'IWMO' for World)
    "EMU": "EZU",  # iShares MSCI Eurozone
}
benchmark = "URTH"  # iShares MSCI World ETF
weights = [0.25, 0.50, 0.25]  # Adjust based on your actual allocation
start_date = "2023-01-01"
end_date = "2024-12-31"

# 2. Download Data
data = yf.download(
    list(tickers.values()) + [benchmark], start=start_date, end=end_date
)["Close"]
returns = data.pct_change().dropna()


# 3. Define the IVOL function
def calculate_ivol(etf_returns, market_returns):
    # Add a constant for the Intercept (Alpha)
    X = sm.add_constant(market_returns)
    model = sm.OLS(etf_returns, X).fit()

    # Residuals represent the return not explained by the market
    residuals = model.resid
    daily_ivol = residuals.std()

    # Annualize (Square root of 252 trading days)
    annualized_ivol = daily_ivol * np.sqrt(252)
    return annualized_ivol, residuals


# 4. Calculate for each ETF
print("--- Individual Idiosyncratic Volatility ---")
all_residuals = {}
for name, ticker in tickers.items():
    ivol, resid = calculate_ivol(returns[ticker], returns[benchmark])
    all_residuals[ticker] = resid
    print(f"{name} ({ticker}): {ivol:.2%}")

# 5. Calculate for the Portfolio
# Create weighted portfolio returns
portfolio_returns = (returns[list(tickers.values())] * weights).sum(axis=1)

portfolio_ivol, p_resid = calculate_ivol(portfolio_returns, returns[benchmark])

print("\n--- Portfolio Level ---")
print(f"Total Portfolio IVOL: {portfolio_ivol:.2%}")

# 6. Calculate Diversification Benefit
# Weighted average of individual IVOLs vs the actual Portfolio IVOL
avg_ivol = sum(
    [
        calculate_ivol(returns[t], returns[benchmark])[0] * w
        for t, w in zip(tickers.values(), weights)
    ]
)
benefit = avg_ivol - portfolio_ivol
print(f"Diversification Benefit (IVOL Reduction): {benefit:.2%}")
