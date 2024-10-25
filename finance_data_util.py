import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
from ml_utils import get_macro_info

# Get total shares (monthly data)
def get_outstanding_share(tick, start):
    try:
        #print('Getting Market Cap...')
        end = datetime.today().strftime('%Y-%m-%d')
        all_months = pd.date_range(start=start, end=end, freq='M')
        if type(tick) == str:
            asset = yf.Ticker(tick)
            shares = asset.get_shares_full(start=(datetime.today() - pd.Timedelta(days=2)).strftime('%Y-%m-%d'), end=(datetime.today().strftime('%Y-%m-%d')))[-1]
        
        elif type(tick) == list:
            shares = pd.DataFrame(dtype='object')
            for ticker in tick:
                asset = yf.Ticker(ticker)
                shares_out = asset.get_shares_full(start=start, end=end)
                shares_out.index = shares_out.index.strftime('%Y-%m')
                shares_out = shares_out.groupby(shares_out.index).last()
                shr = shares_out.reindex(all_months.strftime('%Y-%m')).fillna(method='ffill')
                shares[ticker] = shr
        return shares

    except ValueError as e:
        print(f"Error:{e}")
        print('Input \'string\' or \'list\'' )

# Get top n market cap from file
def get_topN(cap, n):
    top_list = pd.to_numeric(cap.iloc[0], errors='coerce').nlargest(n).index.tolist()
    return top_list

def us_data(mkt, ranking, start, download_start, end, lag, pred_freq='1d'):
    us_stocks = pd.read_csv("us_mktCap_2023_12.csv")
    if type(ranking)==int:
        Top = us_stocks['Symbol'][0:ranking].tolist()
    elif type(ranking)==list:
        Top = us_stocks['Symbol'][ranking].tolist()
    Top.sort()
    top = yf.download(Top, start = start, end = end)
    Top.append(mkt)
    top_all = yf.download(Top, start = start, end = end)
    top_all = top_all.loc[top.index]
    top_shares = get_outstanding_share([item for item in Top if item != mkt], start).fillna(method='bfill')
    macros = get_macro_info(start=download_start, end=end, lag=lag, freq=pred_freq)
    month = yf.download(Top, start = download_start, end = end, interval='1d')

    return top_all, top_shares, macros, month

def tw_data(tw_mkt, ranking, start, download_start, end, lag, pred_freq='1d'): 
    mk_cap_file = r"tw_mktCap.xlsx"
    mkCap = pd.read_excel(mk_cap_file)
    new_columns = {col: col.split(' ')[0] + '.TW' for col in mkCap.columns[1:]}
    mkCap = mkCap.rename(columns=new_columns)
    if type(ranking)==int:
        tw_topN = get_topN(mkCap, ranking)
    elif type(ranking)==list:
        tw_topN = get_topN(mkCap, ranking[1])[ranking[0]-1, ranking[1]]

    tw_top = yf.download(tw_topN, start = start, end = end)
    tw_topN.append(tw_mkt)
    tw_top_all = yf.download(tw_topN, start = start, end = end)
    tw_top_all = tw_top_all.loc[tw_top.index]
    tw_top_shares = get_outstanding_share([item for item in tw_topN if item != tw_mkt], start).fillna(method='bfill')
    tw_macros = get_macro_info(start=download_start, end=end, freq=pred_freq, lag=lag, macro_ind=['^TWII', '00751B.TWO', '00687B.TWO', '00719B.TWO'])
    tw_macros = tw_macros.loc[tw_top.index]
    tw_month = yf.download(tw_topN, start = download_start, end = end, interval='1d')

    return tw_top_all, tw_top_shares, tw_macros, tw_month

def performance_metric(portfolio_df, rf_df):
    trading_days_per_year = 252 
    sharpe_ratios, sortino_ratios, annual_returns = {}, {}, {}
    annual_volatilities, max_drawdowns, daily_vars = {}, {}, {}
    risk_free_rate_df = rf_df#.to_frame('Risk_Free_Rate')

    risk_free_rate_df.index = pd.to_datetime(risk_free_rate_df.index)
    daily_risk_free_rate_df = risk_free_rate_df.resample('D').ffill()

    # Align the risk-free rate series with the portfolio DataFrame's index
    # Assuming portfolio_df index starts at '2023-01-01' and matches the date range
    portfolio_df.index = pd.to_datetime(portfolio_df.index)
    aligned_risk_free_rate = daily_risk_free_rate_df.loc[portfolio_df.index]

    for portfolio_name, portfolio_values in portfolio_df.items():
        portfolio_series = pd.Series(portfolio_values)
        
        # Daily returns
        daily_returns = portfolio_series.pct_change().dropna()
        risk_free_rate = aligned_risk_free_rate.loc[daily_returns.index].values.flatten()/100
        rf = (1 + risk_free_rate)**(1/252) - 1
        # Annual Return
        total_return = (portfolio_series.iloc[-1] - portfolio_series.iloc[0]) / portfolio_series.iloc[0]
        annual_return = (1 + total_return) ** (trading_days_per_year / len(portfolio_series)) - 1
        annual_returns[portfolio_name] = annual_return
        
        # Annual Volatility
        annual_volatility = daily_returns.std() * np.sqrt(trading_days_per_year)
        annual_volatilities[portfolio_name] = annual_volatility
        
        # Sharpe Ratio
        sharpe_ratio = (daily_returns - rf).mean() / daily_returns.std() * np.sqrt(trading_days_per_year)
        sharpe_ratios[portfolio_name] = sharpe_ratio
        
        # Sortino Ratio
        downside_deviation = daily_returns[daily_returns < 0].std()
        sortino_ratio = (daily_returns - rf).mean() / downside_deviation * np.sqrt(trading_days_per_year)
        sortino_ratios[portfolio_name] = sortino_ratio
        
        
        # Maximum Drawdown (MDD)
        cumulative_returns = (1 + daily_returns).cumprod()
        peak = cumulative_returns.cummax()
        drawdown = (cumulative_returns - peak) / peak
        max_drawdown = drawdown.min()
        max_drawdowns[portfolio_name] = max_drawdown
        
        # Daily VaR (Value at Risk)
        confidence_level = 0.95
        daily_var = -np.percentile(daily_returns, 100 * (1 - confidence_level))
        daily_vars[portfolio_name] = daily_var

    # Combine the results into a DataFrame
    results_df = pd.DataFrame({
        'Annual Return': annual_returns,
        'Annual Volatility': annual_volatilities,
        'Sharpe Ratio': sharpe_ratios,
        'Sortino Ratio': sortino_ratios,
        'Max Drawdown': max_drawdowns,
        'Daily VaR (95%)': daily_vars
    })

    # Display the results
    print(results_df)

    return(results_df)

def calculate_portfolio_value(prices_df, weights, initial_value):
    """
    Inputs: Historic stock price data, asset weights
    """
    daily_log_returns = np.log(prices_df / prices_df.shift(1)).fillna(0)
    portfolio_daily_log_returns = daily_log_returns.dot(weights)
    portfolio_value_series = np.exp(portfolio_daily_log_returns.cumsum()) * initial_value

    return portfolio_value_series
