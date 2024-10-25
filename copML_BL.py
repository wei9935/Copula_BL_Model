import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import date, datetime, timedelta
from dateutil.relativedelta import relativedelta

from pypfopt import black_litterman, risk_models, EfficientCVaR, EfficientFrontier
from pypfopt.black_litterman import BlackLittermanModel
import pyfolio as pf
import pandas_ta as ta

import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri
from rpy2.robjects.conversion import py2rpy

from statsmodels.tsa.api import VAR
import mgarch
import backtrader as bt

import warnings

from finance_data_util import * #get_outstanding_share, get_topN, yfInfo_convert_long
from ml_utils import *
from copula_utils import *
from optimize import *


# To suppress all warnings
warnings.filterwarnings("ignore")


def simple_roll_test(df, optimize, rolling_window, forcast_period, mkt, share_df='none'):
    wt_all = pd.DataFrame()
    values = pd.Series(1)
    rolling_window = relativedelta(months = rolling_window)
    roll = relativedelta(months = forcast_period)
    end = df.index[0] + rolling_window
    mons = df.index.strftime('%Y-%m').nunique()

    df = df.loc[:,('Adj Close', slice(None))]
    df.columns = df.columns.get_level_values(1)


    print('Start Roll Testing...')
    for i in range(mons-2):
        start = end - rolling_window
        start_p, end_p = start.strftime('%Y-%m'), end.strftime('%Y-%m')
        print(f'Backtest training data period : {start_p} - {end_p}')
        train = df[
            (df.index.strftime('%Y-%m') >= start.strftime('%Y-%m')) &
            (df.index.strftime('%Y-%m') < end.strftime('%Y-%m'))
            ]
        train_set = train.dropna(axis=1)
        const_col = train_set.columns[train_set.nunique()==1]
        train_set = train_set.drop(columns=const_col)
        prices = train_set.loc[:, train_set.columns != mkt]
        rt = prices.pct_change().dropna(axis=0)
        mu = np.mean(rt)
        
        if optimize == 'EQW' :
            #eqaul weight
            assets = train_set.drop(columns=[mkt])
            wt = {item: 1/len(assets.columns) for item in assets.columns} #eqaul weight
            wt = pd.DataFrame([wt],columns=wt.keys()).T

        elif optimize == 'MKT':
            if type(share_df)==str:
                print('Input share outstanding data')
            else:
                share_month = share_df.copy()
                latest_p = prices.iloc[-1]
                share_month = share_month[prices.columns]
                target_mon = (train_set.index[-1]+relativedelta(month=1)).strftime('%Y-%m')
                share_list = share_month.loc[target_mon]
                mcaps = pd.DataFrame(latest_p*share_list)
                wt = {item: (mcaps.loc[item]/(mcaps.values.sum())).values[0] for item in mcaps.index} #eqaul weight
                wt = pd.DataFrame([wt],columns=wt.keys()).T 
            
        elif optimize == 'MV':
            wt = maxSharpe(mu, rt.cov(), np.mean(mu))
        elif optimize == 'CVaR':
            wt = minCVaR(mu, rt.cov(), sim='Normal')

        wt_all[end.strftime('%Y-%m')] = wt
        test_set = df[
            (df.index.strftime('%Y-%m') >= end.strftime('%Y-%m')) &
            (df.index.strftime('%Y-%m') < (end+roll).strftime('%Y-%m'))
            ]
        test_set = test_set.drop(mkt, axis=1)
        ini_val = values.iloc[-1]
        val = calculate_portfolio_value(test_set, wt, ini_val)
        print(f'Portfolio Value : {val}')
        values = pd.concat([values, val])
        end += roll

        if (end.strftime('%Y-%m') > df.index[-1].strftime('%Y-%m')) == True:
            break
        else:
            continue

    values = values[1:]
    values.index = pd.to_datetime(values.index)

    return values, wt_all

def bl_weight(data, days, outShare, predict, cov_est, eta, market, ml_macro, rf, 
              uncertainty='none', optimize='MV', big_df=None, plot=False):
    ##########################  Data Formating  #################################
    prices = data.loc[:, data.columns != market]
    mkt_price = data[market]
    rt = prices.pct_change().dropna(axis=0)
    mkt_rt = mkt_price.pct_change().dropna()
    latest_p = prices.iloc[-1]

    outShare = outShare[prices.columns]
    target_mon = (data.index[-1]+relativedelta(month=1)).strftime('%Y-%m')
    share_list = outShare.loc[target_mon]
    mcaps = (latest_p*share_list).to_dict() #Dictionary
    ##########################  Prior  #################################
    #print('1. Estimating Covariance Matrix...')
    if type(cov_est) != str:
        cov = cov_est
        cov.columns = prices.columns
        #cov.index = rt.index
    elif cov_est == 'Pearson' :
        cov = rt.cov()

    elif cov_est == 't' :
        cov_est = 'Student t'
        dof = len(rt.columns)-1
        cov = ((dof-2)/dof) * rt.cov()

    elif cov_est == 'Shrink':
        cov = risk_models.CovarianceShrinkage(prices).ledoit_wolf()

    elif cov_est == 'DCC':
        # Fit conditional covariance model : DCC-Garch(1,1)
        print('... Estimating DCC_Garch(1, 1) Covariance Matrix ...')
        dist = 't'
        vol = mgarch.mgarch(dist)
        try:
            vol.fit(rt)
            ndays = days*30 # volatility of nth day
            cov_nextday = vol.predict(ndays)
            cov_nextday['cov']
            cov = pd.DataFrame(cov_nextday['cov'], index=prices.columns, columns=prices.columns)

        except ValueError as e:
            print(e)
            error_strt, error_end = data.index[0].strftime('%Y-%m'), data.index[-1].strftime('%Y-%m')
            print(f'Estimating Covariance with Ledoit-Wolf Shrinkage Method for Period : {error_strt}-{error_end}')
            cov = risk_models.CovarianceShrinkage(prices).ledoit_wolf()

    elif cov_est == 'cop_RV':
        cov, res = cop_cov(prices)
        cov = pd.DataFrame(cov, index=prices.columns, columns=prices.columns)
    
    elif cov_est == 'cop_CV':
        cov, res = cop_cov(prices, 'CVine')
        cov = pd.DataFrame(cov, index=prices.columns, columns=prices.columns)

    elif cov_est == 'cop_DV':
        cov, res = cop_cov(prices, 'Dvine')
        cov = pd.DataFrame(cov, index=prices.columns, columns=prices.columns)

    cov = cov*20
    #print('2. Estimating Equilibrium Retrun...')
    delta = black_litterman.market_implied_risk_aversion(data[market], risk_free_rate=float(rf))
    w_mkt = pd.Series({key: value / sum(mcaps.values()) for key, value in mcaps.items()}).values
    prior_pi = black_litterman.market_implied_prior_returns(mcaps, delta, cov)

    ##########################  Views  #################################
    #print('3. Generating Views...')
    if type(predict) != str:
        view_dict = predict.to_dict()
    
    elif predict == 'VAR': # Predict Price with VAR model
        var_prices = prices.dropna(axis=1)
        var_data = pd.DataFrame(data=[np.log(var_prices[f'{ticker}']/var_prices[f'{ticker}'].shift(1)) for ticker in var_prices.columns]).T
        var_data = var_data.dropna(axis=0)

        modelVAR = VAR(var_data)
        results = modelVAR.fit(1)
        
        lag_order = results.k_ar
        pred_days = len(data)
        views_day = results.forecast(var_data.values[-lag_order:], pred_days)
        views = np.prod((views_day + 1), axis=0) - 1
        view_dict = dict(zip(prices.columns, views.tolist()))

    elif predict == 'RandomForest':
        first = data.index[0].strftime('%Y-%m')
        last = data.index[-1].strftime('%Y-%m')
        ml_macro = ml_macro[(ml_macro.index.strftime('%Y-%m') >= first)&(ml_macro.index.strftime('%Y-%m') <= last)] 
        big_df = big_df[(big_df.index.strftime('%Y-%m') >= first)&(big_df.index.strftime('%Y-%m') <= last)]
        view_dict = {}
        uncertainty=[]
        for tick in prices.columns:
            sub = big_df.loc[:, big_df.columns.get_level_values(1)==tick]
            sub.columns = sub.columns.get_level_values(0)
            view_dict[tick], uncert = rf_predict(sub, ml_macro)
            uncertainty.append(uncert)
        
    elif predict == 'XGB':
        first = data.index[0].strftime('%Y-%m')
        last = data.index[-1].strftime('%Y-%m')
        ml_macro = ml_macro[(ml_macro.index.strftime('%Y-%m') >= first)&(ml_macro.index.strftime('%Y-%m') <= last)] 
        big_df = big_df[(big_df.index.strftime('%Y-%m') >= first)&(big_df.index.strftime('%Y-%m') <= last)]
        view_dict = {}
        uncertainty = []
        for tick in prices.columns:
            sub = big_df.loc[:, big_df.columns.get_level_values(1)==tick]
            sub.columns = sub.columns.get_level_values(0)
            view_dict[tick], uncert = xgb_predict(sub, ml_macro)
            uncertainty.append(uncert)

    ##########################  Posterior caculation  #################################
    #print('4. Calculating Black-Litterman Posterier Returns and Covariance Matrix...')
    if type(uncertainty) != str:
        bl = BlackLittermanModel(cov, pi=prior_pi, absolute_views=view_dict, omega=np.diag(uncertainty))
    else:
        bl = BlackLittermanModel(cov, pi=prior_pi, absolute_views=view_dict)
    
    post_rets = bl.bl_returns()
    post_cov = bl.bl_cov()
    if isinstance(eta, str):
        if eta == False:
            r_dist = 'Normal'
        else:
            raise ValueError("Invalid value for eta. Expected True, False or ndarray")
    else:
        r_dist = np.dot(eta, np.linalg.cholesky(post_cov).T)
        r_dist += np.array(post_rets)
    ##########################  Optimization  #################################
    #print(f'return:\n{post_rets}\ncov = \n{post_cov}')
    #print(eta)
    #print('5. Calculating Optimal Weight...')
    if optimize == 'MV': 
        obj = 'Max Sharpe'
        optimal_weight = maxSharpe(post_rets, post_cov, np.mean(post_rets))

    elif optimize == 'CVaR':
        obj = 'Min CVaR'
        optimal_weight = minCVaR(post_rets, post_cov)

    if plot == True:
        optimal_weight.plot.bar(figsize=(14,6), 
                            title = f'BL with {cov_est} Covariance Matrix ({obj})', grid=True, legend=False);
        plt.ylabel('Percentage')
        plt.show()
    
    pred = pd.DataFrame([view_dict],columns=view_dict.keys()).T

    return optimal_weight

def roll_test(df, predict, cov,  mkt, optimize, rolling_window, forcast_period, 
              ml_macro,  month_data, shares_df, rf_df, nonN=False, uncertainty='none', EQW=False):
    wt_all = pd.DataFrame()
    pred = pd.DataFrame()
    values = pd.Series(1)
    rolling_window = relativedelta(months = rolling_window)
    roll = relativedelta(months = forcast_period)
    end = df.index[0] + rolling_window
    mons = df.index.strftime('%Y-%m').nunique()

    df = df.loc[:,('Adj Close', slice(None))]
    df.columns = df.columns.get_level_values(1)

    share_month = shares_df.copy()
    print('Start Roll Testing...')
    for i in range(mons-2):
        start = end - rolling_window
        start_p, end_p = start.strftime('%Y-%m'), end.strftime('%Y-%m')
        print(f'Backtest training data period : {start_p} - {end_p}')
        train = df[
            (df.index.strftime('%Y-%m') >= start.strftime('%Y-%m')) &
            (df.index.strftime('%Y-%m') < end.strftime('%Y-%m'))
            ]
        train_set = train.dropna(axis=1)
        #train_set = train.loc[:,('Adj Close', slice(None))]
        #train_set.columns = train_set.columns.get_level_values(1)

        # drop unchange value columns
        const_col = train_set.columns[train_set.nunique()==1]
        train_set = train_set.drop(columns=const_col)
        #drop_ticker = train.columns[train.columns.get_level_values(1).isin(const_col)]

        if EQW == True :
            #eqaul weight
            assets = train_set.drop(columns=[mkt])
            wt = {item: 1/len(assets.columns) for item in assets.columns} #eqaul weight
            wt = pd.DataFrame([wt],columns=wt.keys()).T

        elif EQW == 'mkt':
            prices = train_set.loc[:, train_set.columns != mkt]
            latest_p = prices.iloc[-1]
            share_month = share_month[prices.columns]
            target_mon = (train_set.index[-1]+relativedelta(month=1)).strftime('%Y-%m')
            share_list = share_month.loc[target_mon]
            mcaps = pd.DataFrame(latest_p*share_list)
            wt = {item: (mcaps.loc[item]/(mcaps.values.sum())).values[0] for item in mcaps.index} #eqaul weight
            wt = pd.DataFrame([wt],columns=wt.keys()).T 

        else:
            if len(cov) > 10:
                cov_est = read_cop_cov(cov, end.strftime('%Y-%m'))
            else:
                cov_est = cov

            if nonN  == True:
                eta = read_cop_eta(cov, end.strftime('%Y-%m'))
            else:
                eta = nonN

            if type(predict) == str:
                pred = predict
            else:
                pred = predict[end.strftime('%Y-%m')]

            if type(uncertainty) == str:
                unc = uncertainty
                
            else:
                unc = uncertainty[end.strftime('%Y-%m')]

            rf = (1+rf_df.loc[start_p]/100)**(1/252)-1
            wt = bl_weight(train_set, forcast_period, share_month, pred, cov_est, eta, 
                        mkt, ml_macro=ml_macro, rf = rf, uncertainty=unc, optimize=optimize, big_df=month_data)
            
        wt_all[end.strftime('%Y-%m')] = wt
        #remains = train.columns
        test_set = df[
            (df.index.strftime('%Y-%m') >= end.strftime('%Y-%m')) &
            (df.index.strftime('%Y-%m') < (end+roll).strftime('%Y-%m'))
            ]
        test_set = test_set.drop(mkt, axis=1)
        #test_set = test_set[remains]
        ini_val = values.iloc[-1]
        val = calculate_portfolio_value(test_set, wt, ini_val)
        print(f'Portfolio Value : {val}')
        #data_long = yfInfo_convert_long(test_set, mkt) # Reformat data
        #rt, val = back_test_results(wt, data_long, ini_val)
        #rets = pd.concat([rets, rt])
        values = pd.concat([values, val])
        end += roll

        if (end.strftime('%Y-%m') > df.index[-1].strftime('%Y-%m')) == True:
            break
        else:
            continue

    values = values[1:]
    values.index = pd.to_datetime(values.index)

    values.name = f'{cov}_{optimize}'
    return values, wt_all

