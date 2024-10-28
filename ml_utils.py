import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from dateutil.relativedelta import relativedelta
from datetime import datetime

from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.impute import KNNImputer
import pandas_ta as ta

import xgboost as xgb
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

# Create Features
def yf_create_info(df, name='', lag=20):
    stock = df
    stock['y'] = df['Adj Close'].pct_change(lag).shift(lag*(-1))
    stock['return'] = np.log(stock['Adj Close']/stock['Adj Close'].shift(1))
    stock['volume_chg'] = np.log(stock['Volume']/stock['Adj Close'].shift(1))

    #Technical
    ##Momentum
    stock.ta.cci(append=True)
    stock.ta.rsi(append=True)
    stock.ta.roc(append=True)
    stock.ta.stoch(append=True)
    stock.ta.willr(append=True)
    stock.ta.aroon(append=True)
    stock.ta.tsi(append=True)
    #Trend
    stock.ta.sma(append=True)
    stock.ta.ema(append=True)
    stock.ta.macd(append=True)
    stock.ta.adx(append=True)
    stock.ta.t3(append=True)
    #Volume
    stock.ta.obv(append=True)
    stock.ta.mfi(append=True)
    stock.ta.cmf(append=True)
    stock.ta.efi(append=True)
    #Volitility
    stock.ta.bbands(append=True)
    stock.ta.atr(append=True)

    lagged_cols = [col for col in df.columns if col not in df.columns]
    for col in lagged_cols:
        for i in range(1, lag + 1):
            stock[f'{col}_lag_{i}'] = stock[col].shift(i)
           
    for i in range(1, lag):
        stock[f'{i}_day_ret'] = stock['return'].rolling(window=i).sum()
        stock[f'{i}_day_volChg'] = stock['volume_chg'].rolling(window=i).sum()

    if name == '':
        stock = stock
    else:
        stock = stock.add_prefix(f'{name}_')

    stock = stock.loc[:, ~stock.isin([np.inf, -np.inf]).any()]

    return stock.iloc[:, 6:]

def get_macro_info(start, end, freq, lag=20, macro_ind=['^GSPC', '^IRX', '^TNX', '^VIX']):
    df = pd.DataFrame()
    for macro in macro_ind:
        macr_info = yf.download(macro, start = start, end = end, interval=freq)
        macr_info = yf_create_info(macr_info, name=macro, lag=lag)
        df = pd.concat([df, macr_info], axis=1)
    df = df.dropna(axis=1, how='all')
    knn_imputer = KNNImputer(n_neighbors=3)  # You can adjust the number of neighbors (n_neighbors) as needed
    df_imputed = pd.DataFrame(knn_imputer.fit_transform(df), columns=df.columns)
    df_imputed.index = df.index
    return df_imputed

# Predict
def rf_predict(stock_df, macro, lag=20):
    stock_info = yf_create_info(stock_df, lag=lag)
    macro = macro.loc[stock_df.index]
    '''param_grid_rf = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20],
        'min_samples_split': [2, 5]
        }
    fin_df = pd.concat([stock_info, macro], axis=1)
    fin_df = fin_df.rolling(window=5, min_periods=1).mean().fillna(0)
    fin_df = fin_df.fillna(fin_df.rolling(window=5, min_periods=1).mean())
    feats = fin_df.columns
    fin_df['y'] = fin_df['return'].shift(-1)'''
    #tscv = TimeSeriesSplit(n_splits=5)
    full_df = pd.concat([stock_info, macro], axis=1)
    knn_imputer = KNNImputer(n_neighbors=20)
    fin_df = pd.DataFrame(knn_imputer.fit_transform(full_df), columns=full_df.columns)
    fin_df = fin_df.apply(lambda col: col.fillna(col.rolling(window=5, min_periods=1).mean()))
    fin_df.index = full_df.index
    
    X = fin_df.drop('y', axis=1).iloc[:-1]
    y = fin_df['y'].iloc[:-1]
    
    rf_model = RandomForestRegressor(n_estimators=1000, max_depth=10, 
                                     min_samples_split=2, 
                                     random_state=28, 
                                     n_jobs=-1
                                     )
    '''grid_search_rf = GridSearchCV(estimator=rf_model, param_grid=param_grid_rf, cv=tscv, scoring='neg_mean_squared_error')
    grid_search_rf.fit(X, y)

    best_params_rf = grid_search_rf.best_params_
    best_rf_model = RandomForestRegressor(**best_params_rf, random_state=28)
    
    best_rf_model.fit(X, y)'''
    rf_model.fit(X, y)
    pred_set = fin_df[X.columns]
    pred = rf_model.predict(pred_set)[-1]
    mse = mean_squared_error(y, rf_model.predict(X))

    return pred, mse

def xgb_predict(stock_df, macro, lag=20):
    stock_info = yf_create_info(stock_df, lag=lag)
    macro = macro.loc[stock_df.index]
    '''param_grid_xgb = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 6],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 1.0],
        'base_score':0.5, 
        'booster':'gbtree',
        'early_stopping_rounds':50,
        'objective':'reg:absoluteerror',
        'device':'cuda'
    }'''
    #tscv = TimeSeriesSplit(n_splits=5)
    full_df = pd.concat([stock_info, macro], axis=1)
    knn_imputer = KNNImputer(n_neighbors=20)
    fin_df = pd.DataFrame(knn_imputer.fit_transform(full_df), columns=full_df.columns)
    fin_df = fin_df.apply(lambda col: col.fillna(col.rolling(window=5, min_periods=1).mean()))
    fin_df.index = full_df.index
    
    X = fin_df.drop('y', axis=1).iloc[:-1]
    y = fin_df['y'].iloc[:-1]
    X_train, X_val = train_test_split(X, test_size=0.2, shuffle=False)
    y_train, y_val = train_test_split(y, test_size=0.2, shuffle=False)
    '''xgb_model = xgb.XGBRegressor(random_state=28)
    grid_search_xgb = GridSearchCV(estimator=xgb_model, param_grid=param_grid_xgb, cv=tscv, scoring='neg_mean_squared_error')
    grid_search_xgb.fit(X, y)
    best_params_xgb = grid_search_xgb.best_params_

    best_xgb_model = xgb.XGBRegressor(**best_params_xgb, random_state=28)
    best_xgb_model.fit(X, y)'''
    
    xgb_model = xgb.XGBRegressor(base_score=0.5, booster='gbtree', 
                                 n_estimators=1000,
                                 early_stopping_rounds=50,
                                 objective='reg:absoluteerror',
                                 max_depth=3,
                                 learning_rate=0.05,
                                 device='cuda', random_state=123
                                 )
    
    xgb_model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], verbose=100)
    mse = mean_squared_error(y_val, xgb_model.predict(X_val))
    pred_set = fin_df[X.columns]
    pred = xgb_model.predict(pred_set)[-1]

    return pred, mse 

# Read data
def read_predict_data(file):  
    predictions = pd.read_csv(file)
    predictions = predictions.set_index('Unnamed: 0')
    predictions.index.name = None

    return predictions

def ml_perf(pred_df, stock, plot=False):
    pred = pred_df.loc[stock]
    start= datetime.strptime(pred_df.columns[0], '%Y-%m')-relativedelta(months=1)
    start = start.strftime('%Y-%m')+'-01'
    end = (pd.to_datetime(pred_df.columns[-1]) + pd.offsets.MonthEnd(0)).strftime('%Y-%m-%d')
    price = yf.download(stock, start=start, end=end)['Adj Close']
    month_price = price.resample('M').last()
    ret = month_price.pct_change().dropna()
    ret = ret.transpose()
    ret.index=ret.index.strftime('%Y-%m')
    mae = mean_absolute_error(ret, pred)
    mse = mean_squared_error(ret, pred)

    
    if plot:
        fig, ax1 = plt.subplots(2, 1, figsize=(10, 8))

        # Plot monthly returns
        ax1[0].plot(ret, label='Monthly Return', marker='o')
        ax1[0].plot(pred, label='Prediction', marker='x')
        ax1[0].set_title('Prediction vs. Realized Monthly Returns')
        ax1[0].set_xlabel('Month')
        ax1[0].set_ylabel('Monthly Return')
        ax1[0].legend()
        first_month_price = month_price.shift(1).dropna()
        first_month_price.index = pred.index
        predicted_prices = first_month_price * (1 + pred.values)
        last_prices = month_price[1:]
        last_prices.index = predicted_prices.index
        # Plot prices
        ax1[1].plot(last_prices, label='Actual Price', marker='o')
        ax1[1].plot(predicted_prices, label='Predicted Price', marker='x')
        ax1[1].set_title(f'Actual vs. Predicted Prices ({stock}))')
        ax1[1].set_xlabel('Month')
        ax1[1].set_ylabel('Price')
        ax1[1].legend()

        plt.tight_layout()
        plt.show()

    return mae, mse, np.sqrt(mse)

def evaluate_model(pred_df):
    mae_list, mse_list, rmse_list = [], [], []
    for stock in pred_df.index:
        mae, mse, rmse = ml_perf(pred_df, stock)
        mae_list.append(mae)
        mse_list.append(mse)
        rmse_list.append(rmse)
    
    metrics_df = pd.DataFrame({
        'MAE': mae_list,
        'MSE': mse_list,
        'RMSE': rmse_list
    })
    
    return metrics_df

def top_weighted_tickers(df, threshold=0.95):
    result = {}
    for date in df.columns:
        sorted_weights = df[date].sort_values(ascending=False)
        cum_sum = sorted_weights.cumsum()
        top_tickers = sorted_weights[cum_sum <= threshold].index.tolist()
        if cum_sum.iloc[0] > threshold:
            top_tickers.append(cum_sum.index[len(top_tickers)])
        elif cum_sum.iloc[len(top_tickers)] < threshold:
            top_tickers.append(cum_sum.index[len(top_tickers)])
        result[date] = top_tickers
    return result

def calculate_mse(pred_df, wt_df, metric):
    metric_results = {}
    start= datetime.strptime(pred_df.columns[0], '%Y-%m')-relativedelta(months=1)
    start = start.strftime('%Y-%m')+'-01'
    end = (pd.to_datetime(pred_df.columns[-1]) + pd.offsets.MonthEnd(0)).strftime('%Y-%m-%d')

    price = yf.download(list(pred_df.index), start=start, end=end).loc[:, ('Adj Close', slice(None))]
    month_price = price.resample('M').last()
    ret = month_price.pct_change().dropna()
    ret.columns = ret.columns.get_level_values(1)
    ret.index = ret.index.strftime('%Y-%m')
    ret = ret.transpose()

    top_tickers = top_weighted_tickers(wt_df)
    for date, tickers in top_tickers.items():
        pred_values = pred_df.loc[tickers, date]
        real_values = ret.loc[tickers, date]

        if metric == 'MAE':
            error = mean_absolute_error(pred_values, real_values)
        elif metric == 'MSE':
            error = mean_squared_error(pred_values, real_values)

        metric_results[date] = error

    return metric_results

