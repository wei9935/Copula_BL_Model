# Copula Black-Litterman Model with Machine Learning Derived Views and Uncertainty
## Files
* `run_bt.py` : Backtest all stategies.
* `copML_BL.py` : Main framework for calculating weights.
* `copula_utils.py` : Caculate copula covariance matrix, and functions for reading pre-trained data.
* `ml_utils.py` : Machine learning models and feature generation.
* `optimize.py` : Different objective functions for optimizing portfolio.
* `finance_data_util.py` : Tools for fetching financial data and functions for calculating performance of portfolios.
* `./data` : Fundemental and macro datas.
* `./fit_results` : Pre-fit datas for copula and machine learning predictions.
* `./results` : Results for different strategies.
  
## Overview
Enhanced the Black-Litterman model by incorporating vine-copula models for market equilibrium returns and ensemble machine learning for forecasting asset returns. Used ML model errors to quantify view uncertainty, improving portfolio performance and max drawdown in Taiwan’s stock market.

## Data
* Time Period: 2016.01.01 - 2024.03.01
* Stocks:  Top 50 market cap stocks in the US and Taiwan stock market.
* Rebalance Frequency: 1 month
* Data Sorce: Yahoo Finance
   
## Methodology
### Black-Litterman model
* Black-Litterman model is consist of three parts:
 ```
1. Market Equalibrium
2. Personal View
3. Optimization
```
* We use different models in each part of the optimization process to try to enhance the performance.
### Vine-Copula Models
* Captures the dependencies between stocks more accurately.
* We use R package ***'VineCopula'*** to calculate the covariance matrix more efficiently.
* To make this process more smoothly, we use ***'rpy2'*** to run R in a python script.
  
### Ensemble Learning Models
* *Random Forest* and *XGBoost* are used to predict stocks' returns.
* Predict with technical analysis indexs and Macro datas.

### Optimization Objective Functions
* *Max Sharpe Ratio*
* *Minimize CVaR*

### Portfolio Construction
* We test different combinations with different models, shown as below:
  
| **Portfolio Name**      | **Market Equakibrim** | **Views** | **Simulate with Copula Model** | **Machine Learning Uncertainty** |
|---------------|------------------|--------------|---------------------|-------------------------|
| BL-RF         | BL Model     | Random Forest | N                   | N                       |
| Rvine-RF      | R-vine copula    | Random Forest | Y                   | Y                       |
| Cvine-RF      | C-vine copula    | Random Forest | Y                   | Y                       |
| Dvine-RF      | D-vine copula    | Random Forest | Y                   | Y                       |
| BL-XGB        | BL Model     | Xgboost       | N                   | N                       |
| Rvine-XGB     | R-vine copula    | Xgboost       | Y                   | Y                       |
| Cvine-XGB     | C-vine copula    | Xgboost       | Y                   | Y                       |
| Dvine-XGB     | D-vine copula    | Xgboost       | Y                   | Y                       |
  
## Backtest Results-TW
### Taiwan Market 24 Month rolling window 
![tw_bst](result/TW_results_Best%20Performers.png)
### Taiwan Market 60 Month rolling window 
![tw_MLcomp](result/TW_results_Compare%20of%20Different%20Predictor.png)
### Taiwan Portfolio Performances
| **Strategy**                             | **Annual Return** | **Annual Volatility** | **Sharpe Ratio** | **Sortino Ratio** | **Max Drawdown** | **Daily VaR** |
|------------------------------------------|-------------------|-----------------------|------------------|-------------------|------------------|---------------|
| **Panel A: Max Sharpe ratio**             |                   |                       |                  |                   |                  |               |
| BL-RF                                     | 11.51%            | 10.39%                | 1.02             | 1.32              | -13.86%          | 0.99%         |
| Rvine-RF                                  | ***30.74%***            | 19.85%                | 1.41             | 1.97              | -24.25%          | 1.99%         |
| BL-XGB                                    | 19.77%            | 12.14%                | ***1.48***           | 2.04              | -16.79%          | 1.17%         |
| Rvine-XGB                                 | ***31.32%***           | 19.91%                | ***1.43***             | 2.00              | -24.23%          | 2.00%         |
| **Panel B: Min CVaR**                      |                   |                       |                  |                   |                  |               |
| BL-RF                                     | 24.85%            | 21.92%                | 1.08             | 1.28              | -26.36%          | 1.40%         |
| Rvine-RF                                  | 24.07%            | 21.08%                | 1.09             | 1.69              | -28.87%          | 2.12%         |
| BL-XGB                                    | ***40.85%***            | 23.64%                | 1.53             | ***1.92***              | -24.33%          | 1.63%         |
| Rvine-XGB                                 | 24.59%            | 21.49%                | 1.09             | 1.69              | -30.14%          | 2.10%         |
| **Panel C: Copula Simulated Minimum CVaR** |                   |                       |                  |                   |                  |               |
| R-vine RF                                 | 25.56%            | 21.15%                | 1.14             | 1.77              | -27.52%          | 2.09%         |
| R-vine XGB                                | 26.67%            | 21.34%                | 1.18             | 1.84              | -29.49%          | 2.05%         |
| **Panel D: Machine Learning Uncertainty** |                   |                       |                  |                   |                  |               |
| R-vine RF                                 | ***32.39%***            | 22.43%                | 1.33             | 1.84              | -24.38%          | 2.32%         |
| R-vine XGB                                | ***34.39%***            | 21.97%                | 1.42             | ***1.98***              | -24.65%          | 2.24%         |
| **Panel E: Hybrid Model Portfolio**                  |                   |                       |                  |                   |                  |               |
| R-vine RF                                 | 30.80%            | 22.31%                | 1.28             | 1.77              | -25.65%          | 2.22%         |
| R-vine XGB                                | ***36.40%***            | 21.90%                | 1.49             | ***2.09***              | -25.16%          | 2.28%         |
| **Panel F: Benchmarks**                       |                   |                       |                  |                   |                  |               |
| market weight                             | 13.69%            | 18.94%                | 0.73             | 1.05              | -27.40%          | 1.86%         |
| equal weight                              | 16.79%            | 16.58%                | 0.97             | 1.18              | -28.07%          | 1.57%         |
| max Sharpe                                | 16.63%            | 11.43%                | 1.33             | 1.73              | -16.80%          | 1.06%         |
| min CVaR                                  | 10.48%            | 9.46%                 | 1.01             | 1.32              | -13.39%          | 0.90%         |

## Backtest Result-US
### Us Market 24 Month rolling window 
![us_bst](result/US_results_Best%20Performers.png)
### Taiwan Market 60 Month rolling window 
![us_MLcomp](result/US_results_Compare%20of%20Different%20Predictor.png)
### 
| **Strategy**     | **Annual Return** | **Annual Volatility** | **Sharpe Ratio** | **Sortino Ratio** | **Max Drawdown** | **Daily VaR** |
|------------------|-------------------|-----------------------|------------------|-------------------|------------------|---------------|
| **Panel A: Max Sharpe ratio** |                   |                       |                  |                   |                  |               |
| BL-RF            | 7.51%             | 16.88%                | 0.38             | 0.44              | -28.63%          | 1.46%         |
| Rvine-RF         | 14.69%            | 23.10%                | 0.61             | 0.77              | -31.09%          | 2.17%         |
| BL-XGB           | 10.26%            | 17.98%                | 0.50             | 0.60              | -30.59%          | 1.60%         |
| Rvine-XGB        | ***16.79%***           | 23.17%                | 0.69             | ***0.87***              | -33.36%          | 2.19%         |
 **Panel B: Min CVaR**         |                   |                       |                  |                   |                  |               |
| BL-RF            | 11.35%            | 18.14%                | 0.56             | 0.63              | -30.22%          | 1.64%         |
| Rvine-RF         | 18.47%            | 23.19%                | 0.75             | 0.96              | -31.46%          | 2.20%         |
| BL-XGB           | 10.76%            | 21.23%                | 0.48             | 0.53              | -36.80%          | 1.81%         |
| Rvine-XGB        | ***20.20%***            | 23.46%                | 0.80             | ***1.03***              | -32.10%          | 2.22%         |
| **Panel C: Copula Simulated Minimum CVaR** |                   |                       |                  |                   |                  |               |
| R-vine RF            | 18.46%            | 23.41%                | 0.74             | 0.96              | -31.04%          | 2.17%         |
| R-vine XGB           | ***21.06%***            | 23.58%                | ***0.83***             | ***1.05***              | -33.03%          | 2.23%         |
| **Panel D: Machine Learning Uncertainty**   |                   |                       |                  |                   |                  |               |
| R-vine RF            | 17.30%            | 24.15%                | 0.69             | 0.88              | -33.78%          | 2.32%         |
| R-vine XGB           | 16.29%            | 24.18%                | 0.65             | 0.84              | -34.76%          | 2.31%         |
| **Panel E: Hybrid Model**           |                   |                       |                  |                   |                  |               |
| R-vine RF            | 18.41%            | 24.03%                | 0.73             | 0.93              | -33.19%          | 2.28%         |
| R-vine XGB           | 15.81%            | 24.03%                | 0.63             | 0.82              | -33.74%          | 2.26%         |
| **Panel F: Benchmarks**           |                   |                       |                  |                   |                  |               |
| market weight    | 10.29%            | 23.40%                | 0.44             | 0.55              | -33.40%          | 2.22%         |
| equal weight     | 12.90%            | 21.66%                | 0.56             | 0.69              | -32.57%          | 1.86%         |
| max Sharpe       | 6.38%             | 17.07%                | 0.31             | 0.37              | -29.90%          | 1.54%         |
| min CVaR         | 5.68%             | 16.73%                | 0.27             | 0.32              | -29.26%          | 1.46%         |
