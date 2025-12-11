# 📊 CODS 626 – Financial Derivatives & Risk Management

## Term Project: Securitization & the 2007–2009 Credit Crisis

This repository contains the materials submitted for the **CODS 626 (Financial Derivatives & Risk Management)** final term project.  
The project investigates how securitization of subprime mortgages affected the risk and stock-market performance** of major U.S. financial institutions during the 2007–2009 financial crisis.  
The analysis compares high-exposure banks with lower-exposure banks over the period 2003–2009, using Python to compute volatility, beta, drawdowns, correlation structures, and statistical validation tests.

---

## 📁 Contents

- **data/**  
  Contains the processed dataset files used in the analysis  
  *(monthly returns, pre-crisis and crisis splits, bank vs market data).*

- **results/**  
  Includes all generated outputs such as:  
  – Volatility & beta comparison plots  
  – Maximum drawdown charts  
  – Correlation heatmaps  
  – Statistical test summaries  
  – Regression output

- **securitization_data.py**  
  Python script for downloading and preprocessing historical price data (2003–2009).

- **securitization_analysis.py**  
  Python script performing the full analysis: risk metrics, statistical tests, and figure generation.

---

## 📝 Notes

This repository is intended solely as documentation and backup for the course submission for **CODS 626 – Financial Derivatives & Risk Management**.  
All analysis was conducted using publicly available market data via the `yfinance` API.

