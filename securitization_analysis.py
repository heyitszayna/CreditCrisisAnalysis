""" Analysis Script """

import pandas as pd
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
from scipy import stats
import statsmodels.api as sm


HIGH_EXPOSURE = ['C', 'MS', 'BAC']
LOW_EXPOSURE  = ['JPM', 'WFC', 'GS']
BANK_TICKERS  = HIGH_EXPOSURE + LOW_EXPOSURE
MARKET_TICKER = '^GSPC'

PRE_CRISIS_START   = "2003-01-01"
PRE_CRISIS_END     = "2006-12-31"
CRISIS_START       = "2007-01-01"
CRISIS_END         = "2009-12-31"

DATA_DIR    = Path("data")
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)


# Helper functions
def compute_beta(stock_returns: pd.Series, market_returns: pd.Series) -> float:
    """ Compute CAPM beta of a stock vs the market using covariance/variance. """

    # Align on common dates and drop NaNs
    df = pd.concat([stock_returns, market_returns], axis=1).dropna()
    if df.shape[0] < 2:
        return np.nan
    ri = df.iloc[:, 0]
    rm = df.iloc[:, 1]
    cov = np.cov(ri, rm, ddof=1)[0, 1]
    var_m = np.var(rm, ddof=1)
    if var_m == 0:
        return np.nan
    return cov / var_m


def max_drawdown(price_series: pd.Series) -> float:
    """ Compute maximum drawdown for a given price series. """

    if price_series.dropna().empty:
        return np.nan
    cum_max = price_series.cummax()
    drawdown = price_series / cum_max - 1.0
    return drawdown.min()  # most negative drawdown


def annualize_vol(monthly_vol):
    """ Annualize monthly volatility (assuming 12 periods per year). """

    return monthly_vol * np.sqrt(12)


def build_equal_weight_portfolio(returns_df: pd.DataFrame) -> pd.Series:
    """ Build an equal-weighted portfolio from a DataFrame of returns. """

    return returns_df.mean(axis=1, skipna=True)


def load_data():
    """ Load all necessary data from the data/ folder. """

    monthly_returns_full = pd.read_csv(
        DATA_DIR / "monthly_returns_full_sample.csv",
        index_col=0,
        parse_dates=True
    )

    monthly_prices = pd.read_csv(
        DATA_DIR / "monthly_prices.csv",
        index_col=0,
        parse_dates=True
    )

    pre_crisis_returns = pd.read_csv(
        DATA_DIR / "monthly_returns_pre_crisis.csv",
        index_col=0,
        parse_dates=True
    )

    crisis_returns = pd.read_csv(
        DATA_DIR / "monthly_returns_crisis.csv",
        index_col=0,
        parse_dates=True
    )

    # Separate banks and market for full sample
    bank_returns_full = monthly_returns_full[[c for c in monthly_returns_full.columns if c in BANK_TICKERS]].copy()
    market_returns_full = monthly_returns_full[MARKET_TICKER].copy()

    return (monthly_returns_full,
            monthly_prices,
            pre_crisis_returns,
            crisis_returns,
            bank_returns_full,
            market_returns_full)


def compute_metrics(bank_returns: pd.DataFrame,
                    market_returns: pd.Series,
                    monthly_prices: pd.DataFrame):
    """ Compute metrics for each bank and for each period (pre-crisis, crisis). """

    # Masks for periods
    idx = bank_returns.index
    pre_mask = (idx >= PRE_CRISIS_START) & (idx <= PRE_CRISIS_END)
    crisis_mask = (idx >= CRISIS_START) & (idx <= CRISIS_END)

    # Set up container for metrics
    rows = []
    for ticker in BANK_TICKERS:
        if ticker not in bank_returns.columns:
            continue

        # Full series for the bank
        r_full = bank_returns[ticker]

        # Period splits
        r_pre = r_full[pre_mask]
        r_crisis = r_full[crisis_mask]

        # Prices for drawdown (use monthly prices)
        if ticker in monthly_prices.columns:
            p_full = monthly_prices[ticker]
        else:
            p_full = None

        # Market returns for periods
        m_pre = market_returns[pre_mask]
        m_crisis = market_returns[crisis_mask]

        # Compute metrics
        metrics = {
            "Ticker": ticker,
            # Pre-crisis
            "mean_return_pre": r_pre.mean(),
            "vol_pre": r_pre.std(ddof=1),
            "beta_pre": compute_beta(r_pre, m_pre),
            # Crisis
            "mean_return_crisis": r_crisis.mean(),
            "vol_crisis": r_crisis.std(ddof=1),
            "beta_crisis": compute_beta(r_crisis, m_crisis),
            # Full period
            "mean_return_full": r_full.mean(),
            "vol_full": r_full.std(ddof=1),
            "beta_full": compute_beta(r_full, market_returns),
            "max_drawdown_full": max_drawdown(p_full) if p_full is not None else np.nan
        }

        # Annualized vols
        metrics["ann_vol_pre"] = annualize_vol(metrics["vol_pre"])
        metrics["ann_vol_crisis"] = annualize_vol(metrics["vol_crisis"])
        metrics["ann_vol_full"] = annualize_vol(metrics["vol_full"])

        rows.append(metrics)

    metrics_df = pd.DataFrame(rows)
    metrics_df.set_index("Ticker", inplace=True)

    # Save to CSV
    metrics_df.to_csv(RESULTS_DIR / "bank_metrics.csv")

    return metrics_df


def group_portfolios_and_tests(bank_returns: pd.DataFrame,
                               market_returns: pd.Series):
    """ Build equal-weighted portfolios for all banks. """

    # Ensure we only use available tickers
    high_banks = [t for t in HIGH_EXPOSURE if t in bank_returns.columns]
    low_banks = [t for t in LOW_EXPOSURE if t in bank_returns.columns]

    high_returns = bank_returns[high_banks]
    low_returns = bank_returns[low_banks]

    high_portfolio = build_equal_weight_portfolio(high_returns)
    low_portfolio = build_equal_weight_portfolio(low_returns)

    # Period masks
    idx = bank_returns.index
    pre_mask = (idx >= PRE_CRISIS_START) & (idx <= PRE_CRISIS_END)
    crisis_mask = (idx >= CRISIS_START) & (idx <= CRISIS_END)

    # Split series
    hp_pre = high_portfolio[pre_mask].dropna()
    hp_crisis = high_portfolio[crisis_mask].dropna()

    lp_pre = low_portfolio[pre_mask].dropna()
    lp_crisis = low_portfolio[crisis_mask].dropna()

    # Compute basic group metrics
    def group_metrics(series_pre, series_crisis, label):
        return {
            "group": label,
            "mean_return_pre": series_pre.mean(),
            "mean_return_crisis": series_crisis.mean(),
            "vol_pre": series_pre.std(ddof=1),
            "vol_crisis": series_crisis.std(ddof=1),
            "ann_vol_pre": annualize_vol(series_pre.std(ddof=1)),
            "ann_vol_crisis": annualize_vol(series_crisis.std(ddof=1))
        }

    high_metrics = group_metrics(hp_pre, hp_crisis, "HighExposure")
    low_metrics  = group_metrics(lp_pre, lp_crisis, "LowExposure")

    group_results = pd.DataFrame([high_metrics, low_metrics]).set_index("group")
    group_results.to_csv(RESULTS_DIR / "group_portfolio_metrics.csv")

    # Statistical tests
    test_results = {}

    # t-test for means (pre vs crisis) for each group
    t_high = stats.ttest_ind(hp_pre, hp_crisis, equal_var=False, nan_policy='omit')
    t_low  = stats.ttest_ind(lp_pre, lp_crisis, equal_var=False, nan_policy='omit')

    test_results["t_test_high_pre_vs_crisis"] = {"statistic": t_high.statistic, "pvalue": t_high.pvalue}
    test_results["t_test_low_pre_vs_crisis"]  = {"statistic": t_low.statistic,  "pvalue": t_low.pvalue}

    # Variance test (Levene) for each group (pre vs crisis)
    lev_high = stats.levene(hp_pre, hp_crisis, center='mean')
    lev_low  = stats.levene(lp_pre, lp_crisis, center='mean')

    test_results["levene_var_test_high"] = {"statistic": lev_high.statistic, "pvalue": lev_high.pvalue}
    test_results["levene_var_test_low"]  = {"statistic": lev_low.statistic,  "pvalue": lev_low.pvalue}

    # Simple regression: portfolio return on market return + crisis dummy (for high exposure portfolio)
    crisis_dummy = pd.Series(0, index=idx)
    crisis_dummy.loc[idx >= CRISIS_START] = 1

    # Align series
    reg_df = pd.DataFrame({
        "high_portfolio": high_portfolio,
        "market": market_returns,
        "crisis_dummy": crisis_dummy
    }).dropna()

    X = pd.DataFrame({
        "market": reg_df["market"],
        "crisis_dummy": reg_df["crisis_dummy"]
    })
    X = sm.add_constant(X)  # add intercept
    y = reg_df["high_portfolio"]

    model = sm.OLS(y, X).fit()
    test_results["regression_high_portfolio"] = {
        "params": model.params.to_dict(),
        "pvalues": model.pvalues.to_dict(),
        "r_squared": model.rsquared
    }

    # Save test results as JSON-like text
    test_results_df = pd.DataFrame(test_results).T
    test_results_df.to_csv(RESULTS_DIR / "statistical_tests_summary.csv")

    # Also save full regression summary as a text file
    with open(RESULTS_DIR / "regression_high_portfolio_summary.txt", "w") as f:
        f.write(model.summary().as_text())

    return group_results, test_results


# Plots
def plot_price_history(monthly_prices: pd.DataFrame):
    """ Plot price history for all banks. """

    plt.figure(figsize=(10, 6))
    for ticker in BANK_TICKERS:
        if ticker in monthly_prices.columns:
            plt.plot(monthly_prices.index, monthly_prices[ticker], label=ticker)

    plt.title("Monthly Prices of Major Financial Institutions (2003–2009)")
    plt.xlabel("Date")
    plt.ylabel("Price (Adj Close)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "price_history_banks.png", dpi=300)
    plt.close()


def plot_monthly_returns(bank_returns: pd.DataFrame):
    """ Plot monthly returns for banks (all together, but this can be noisy). """

    plt.figure(figsize=(10, 6))
    for ticker in BANK_TICKERS:
        if ticker in bank_returns.columns:
            plt.plot(bank_returns.index, bank_returns[ticker], label=ticker, alpha=0.7)

    plt.axhline(0, color='black', linewidth=0.8)
    plt.title("Monthly Stock Returns of Major Financial Institutions")
    plt.xlabel("Date")
    plt.ylabel("Monthly Return")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "monthly_returns_banks.png", dpi=300)
    plt.close()


def plot_volatility_comparison(metrics_df: pd.DataFrame):
    """ Bar chart: pre-crisis vs crisis annualized volatility per bank. """

    # Use annualized vol for readability
    x = np.arange(len(metrics_df.index))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, metrics_df["ann_vol_pre"], width, label="Pre-crisis")
    plt.bar(x + width/2, metrics_df["ann_vol_crisis"], width, label="Crisis")

    plt.xticks(x, metrics_df.index)
    plt.ylabel("Annualized Volatility")
    plt.title("Annualized Volatility: Pre-crisis vs Crisis")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "volatility_comparison.png", dpi=300)
    plt.close()


def plot_beta_comparison(metrics_df: pd.DataFrame):
    """ Bar chart: pre-crisis vs crisis beta per bank. """

    x = np.arange(len(metrics_df.index))
    width = 0.35

    plt.figure(figsize=(10, 6))
    plt.bar(x - width/2, metrics_df["beta_pre"], width, label="Pre-crisis")
    plt.bar(x + width/2, metrics_df["beta_crisis"], width, label="Crisis")

    plt.xticks(x, metrics_df.index)
    plt.ylabel("Beta vs S&P 500")
    plt.title("Beta: Pre-crisis vs Crisis")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "beta_comparison.png", dpi=300)
    plt.close()


def plot_correlation_heatmaps(pre_returns: pd.DataFrame, crisis_returns: pd.DataFrame):
    """ Plot correlation heatmaps for pre-crisis and crisis periods (banks only). """

    pre_corr = pre_returns[BANK_TICKERS].corr()
    crisis_corr = crisis_returns[BANK_TICKERS].corr()

    # Pre-crisis heatmap
    plt.figure(figsize=(6, 5))
    plt.imshow(pre_corr, interpolation='nearest')
    plt.colorbar(label='Correlation')
    plt.xticks(range(len(pre_corr.columns)), pre_corr.columns, rotation=45)
    plt.yticks(range(len(pre_corr.index)), pre_corr.index)
    plt.title("Correlation Matrix (Pre-crisis)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "correlation_matrix_pre_crisis.png", dpi=300)
    plt.close()

    # Crisis heatmap
    plt.figure(figsize=(6, 5))
    plt.imshow(crisis_corr, interpolation='nearest')
    plt.colorbar(label='Correlation')
    plt.xticks(range(len(crisis_corr.columns)), crisis_corr.columns, rotation=45)
    plt.yticks(range(len(crisis_corr.index)), crisis_corr.index)
    plt.title("Correlation Matrix (Crisis)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "correlation_matrix_crisis.png", dpi=300)
    plt.close()


def plot_max_drawdown(metrics_df: pd.DataFrame):
    """ Horizontal bar chart of max drawdown per bank. """

    plt.figure(figsize=(8, 5))
    dd = metrics_df["max_drawdown_full"]
    plt.barh(dd.index, dd.values)
    plt.xlabel("Maximum Drawdown")
    plt.title("Maximum Drawdown Over Full Period (2003–2009)")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "max_drawdown_banks.png", dpi=300)
    plt.close()


# Main execution
def main():
    print("Loading data...")
    (monthly_returns_full,
     monthly_prices,
     pre_crisis_returns,
     crisis_returns,
     bank_returns_full,
     market_returns_full) = load_data()

    print("Computing bank-level metrics...")
    metrics_df = compute_metrics(bank_returns_full, market_returns_full, monthly_prices)
    print("Bank-level metrics saved to results/bank_metrics.csv")
    print(metrics_df)

    print("Building group portfolios and running tests...")
    group_results, test_results = group_portfolios_and_tests(bank_returns_full, market_returns_full)
    print("Group portfolio metrics saved to results/group_portfolio_metrics.csv")
    print(group_results)
    print("Statistical test summary saved to results/statistical_tests_summary.csv")

    print("Generating plots...")
    # Bank-level
    plot_price_history(monthly_prices)
    plot_monthly_returns(bank_returns_full)
    plot_volatility_comparison(metrics_df)
    plot_beta_comparison(metrics_df)
    plot_correlation_heatmaps(pre_crisis_returns, crisis_returns)
    plot_max_drawdown(metrics_df)

    print("All plots saved in the 'results/' folder.")
    print("Done.")


if __name__ == "__main__":
    main()
