""" Data Download Script """

import pandas as pd
import yfinance as yf
from pathlib import Path

# Bank tickers
HIGH_EXPOSURE = ['C', 'MS', 'BAC']     # Citigroup, Morgan Stanley, Bank of America
LOW_EXPOSURE  = ['JPM', 'WFC', 'GS']   # JPMorgan, Wells Fargo, Goldman Sachs
BANK_TICKERS  = HIGH_EXPOSURE + LOW_EXPOSURE

# Market index (S&P 500)
MARKET_TICKER = '^GSPC'

# Date range (full sample)
START_DATE = "2003-01-01"
END_DATE   = "2009-12-31"

# Time windows
PRE_CRISIS_START   = "2003-01-01"
PRE_CRISIS_END     = "2006-12-31"
CRISIS_START       = "2007-01-01"
CRISIS_END         = "2009-12-31"

# Output folder
DATA_DIR = Path("data")
DATA_DIR.mkdir(exist_ok=True)

# Helper functions 

def download_price_data():
    """ Download daily Adjusted Close prices for all banks and the market index. """

    tickers = BANK_TICKERS + [MARKET_TICKER]

    # Download all at once; auto_adjust=False so we explicitly use 'Adj Close'
    raw = yf.download(
        tickers,
        start=START_DATE,
        end=END_DATE,
        auto_adjust=False,
        progress=False
    )

    # Keep only Adjusted Close
    if 'Adj Close' not in raw.columns:
        raise ValueError("Expected 'Adj Close' in downloaded data columns.")

    prices = raw['Adj Close'].copy()

    # Drop days where everything is NaN
    prices = prices.dropna(how="all")

    # Save raw prices
    prices.to_csv(DATA_DIR / "daily_adj_close_prices.csv", index=True)

    return prices


def compute_monthly_returns(prices: pd.DataFrame):
    """ Convert daily prices to monthly returns for each ticker. """

    # Month-end prices (last available trading day in each month)
    monthly_prices = prices.resample("M").last()

    # Percentage change month to month
    monthly_returns = monthly_prices.pct_change().dropna(how="all")

    # Save
    monthly_prices.to_csv(DATA_DIR / "monthly_prices.csv", index=True)
    monthly_returns.to_csv(DATA_DIR / "monthly_returns_full_sample.csv", index=True)

    return monthly_returns


def split_time_windows(monthly_returns: pd.DataFrame):
    """ Split the monthly returns into pre-crisis and crisis periods. """

    idx = monthly_returns.index

    pre_mask = (idx >= PRE_CRISIS_START) & (idx <= PRE_CRISIS_END)
    crisis_mask = (idx >= CRISIS_START) & (idx <= CRISIS_END)

    pre_crisis_returns = monthly_returns.loc[pre_mask].copy()
    crisis_returns = monthly_returns.loc[crisis_mask].copy()

    # Save
    pre_crisis_returns.to_csv(DATA_DIR / "monthly_returns_pre_crisis.csv", index=True)
    crisis_returns.to_csv(DATA_DIR / "monthly_returns_crisis.csv", index=True)

    return pre_crisis_returns, crisis_returns


def separate_banks_and_market(monthly_returns: pd.DataFrame):
    """ Split monthly returns into bank_returns and market_returns. """

    # Some tickers might be missing (e.g., MER shorter history); intersect to be safe
    available_cols = monthly_returns.columns.tolist()
    banks_present = [t for t in BANK_TICKERS if t in available_cols]

    bank_returns = monthly_returns[banks_present].copy()

    if MARKET_TICKER not in monthly_returns.columns:
        raise ValueError(f"Market ticker {MARKET_TICKER} not found in monthly returns.")

    market_returns = monthly_returns[MARKET_TICKER].copy()

    # Save separately
    bank_returns.to_csv(DATA_DIR / "monthly_bank_returns_full_sample.csv", index=True)
    market_returns.to_csv(DATA_DIR / "monthly_market_returns_full_sample.csv", index=True)

    return bank_returns, market_returns


# Main execution
def main():
    print("Downloading daily price data...")
    prices = download_price_data()
    print(f"Downloaded daily prices with shape: {prices.shape}")

    print("Computing monthly returns...")
    monthly_returns = compute_monthly_returns(prices)
    print(f"Monthly returns shape: {monthly_returns.shape}")

    print("Splitting into pre-crisis and crisis windows...")
    pre_crisis_returns, crisis_returns = split_time_windows(monthly_returns)
    print(f"Pre-crisis returns shape: {pre_crisis_returns.shape}")
    print(f"Crisis returns shape: {crisis_returns.shape}")

    print("Separating bank and market returns...")
    bank_returns, market_returns = separate_banks_and_market(monthly_returns)
    print(f"Bank returns columns: {bank_returns.columns.tolist()}")
    print(f"Market returns length: {len(market_returns)}")

    print("All data saved in the 'data/' folder.")


if __name__ == "__main__":
    main()
