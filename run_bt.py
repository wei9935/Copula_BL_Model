from copML_BL import *
##cov : str or df of dfs (path or df) /eg. r'C:\Users\User\Desktop\NTHU\Thesis\Results\18-24_TW_TOP50_Cvine_cov.json'
##predict : str or df (df only)       /eg. r'C:\Users\User\Desktop\NTHU\Thesis\Results\18-24_TW_TOP50_RF.csv'

time_period = '18-24'

t50cov_Rv = f'fit_result/{time_period}_TW_TOP50_Rvine_cov.json'

t50RF = read_predict_data(f'fit_result/{time_period}_twTOP50_RF.csv')
t50XGB =  read_predict_data(f'fit_result/{time_period}_twTOP50_XGB.csv')
t50RF_un = read_predict_data(f'fit_result/{time_period}_twTOP50_RF_unc.csv')
t50XGB_un =  read_predict_data(f'fit_result/{time_period}_twTOP50_XGB_unc.csv')

u50cov_Rv = f'fit_result/{time_period}_US_TOP50_Rvine_cov.json'

u50RF = read_predict_data(f'fit_result/{time_period}_usTOP50_RF.csv')
u50XGB =  read_predict_data(f'fit_result/{time_period}_usTOP50_XGB.csv')
u50RF_un = read_predict_data(f'fit_result/{time_period}_usTOP50_RF_unc.csv')
u50XGB_un =  read_predict_data(f'fit_result/{time_period}_usTOP50_XGB_unc.csv')

##### TW #####
start = '2019-01-01'
download_start = (pd.to_datetime(start) - relativedelta(months=1)).replace(day=1).strftime('%Y-%m-%d')
end = '2024-03-01'
pred_freq = '1d'
trade_period = f"{str((pd.to_datetime(start) + relativedelta(years=1)).year)[-2:]}-{str(pd.to_datetime(end).year)[-2:]}"

train_length = 12  #Month
hold_period = 1    #Month
tw_mkt = '^TWII'
tw_top_all, tw_top_shares, tw_macros, tw_month = tw_data(tw_mkt=tw_mkt, ranking=50, start=start, download_start=download_start, end=end, lag=20)
tw_rf = pd.read_csv('data/TW_risk_free_rate.csv', index_col=0)

# max sharpe
t50_val_RF, t50_wt_RF = roll_test(tw_top_all, t50RF, 'Pearson', tw_mkt, optimize='MV', rolling_window=train_length, forcast_period=hold_period, 
                                  ml_macro=tw_macros, month_data=tw_month, shares_df=tw_top_shares, rf_df=tw_rf)
t50_val_RFS, t50_wt_RFS = roll_test(tw_top_all, t50RF, 'Shrink', tw_mkt, optimize='MV', rolling_window=train_length, forcast_period=hold_period, 
                                    ml_macro=tw_macros, month_data=tw_month, shares_df=tw_top_shares, rf_df=tw_rf)
t50_val_rvRF, t50_wt_rvRF = roll_test(tw_top_all, t50RF, t50cov_Rv, tw_mkt, optimize='MV', rolling_window=train_length, forcast_period=hold_period, 
                                      ml_macro=tw_macros, month_data=tw_month, shares_df=tw_top_shares, rf_df=tw_rf)

t50_val_XG, t50_wt_XG = roll_test(tw_top_all, t50XGB, 'Pearson', tw_mkt, optimize='MV', rolling_window=train_length, forcast_period=hold_period, 
                                  ml_macro=tw_macros, month_data=tw_month, shares_df=tw_top_shares, rf_df=tw_rf)
t50_val_XGS, t50_wt_XGS = roll_test(tw_top_all, t50XGB, 'Shrink', tw_mkt, optimize='MV', rolling_window=train_length, forcast_period=hold_period, 
                                    ml_macro=tw_macros, month_data=tw_month, shares_df=tw_top_shares, rf_df=tw_rf)
t50_val_rvXG, t50_wt_rvXG = roll_test(tw_top_all, t50XGB, t50cov_Rv, tw_mkt, optimize='MV', rolling_window=train_length, forcast_period=hold_period, 
                                      ml_macro=tw_macros, month_data=tw_month, shares_df=tw_top_shares, rf_df=tw_rf)

# min CVaR
cvt50_val_RF, cvt50_wt_RF = roll_test(tw_top_all, t50RF, 'Pearson', tw_mkt, optimize='CVaR', rolling_window=train_length, forcast_period=hold_period, 
                                      ml_macro=tw_macros, month_data=tw_month, shares_df=tw_top_shares, rf_df=tw_rf)
cvt50_val_RFS, cvt50_wt_RFS = roll_test(tw_top_all, t50RF, 'Shrink', tw_mkt, optimize='CVaR', rolling_window=train_length, forcast_period=hold_period, 
                                        ml_macro=tw_macros, month_data=tw_month, shares_df=tw_top_shares, rf_df=tw_rf)
cvt50_val_rvRF, cvt50_wt_rvRF = roll_test(tw_top_all, t50RF, t50cov_Rv, tw_mkt, optimize='CVaR', rolling_window=train_length, forcast_period=hold_period, 
                                          ml_macro=tw_macros, month_data=tw_month, shares_df=tw_top_shares, rf_df=tw_rf)

cvt50_val_XG, cvt50_wt_XG = roll_test(tw_top_all, t50XGB, 'Pearson', tw_mkt, optimize='CVaR', rolling_window=train_length, forcast_period=hold_period, 
                                      ml_macro=tw_macros, month_data=tw_month, shares_df=tw_top_shares, rf_df=tw_rf)
cvt50_val_XGS, cvt50_wt_XGS = roll_test(tw_top_all, t50XGB, 'Shrink', tw_mkt, optimize='CVaR', rolling_window=train_length, forcast_period=hold_period, 
                                        ml_macro=tw_macros, month_data=tw_month, shares_df=tw_top_shares, rf_df=tw_rf)
cvt50_val_rvXG, cvt50_wt_rvXG = roll_test(tw_top_all, t50XGB, t50cov_Rv, tw_mkt, optimize='CVaR', rolling_window=train_length, forcast_period=hold_period, 
                                          ml_macro=tw_macros, month_data=tw_month, shares_df=tw_top_shares, rf_df=tw_rf)

# Non-Normal
# Max Shapre
Et50_val_rvRF, Et50_wt_rvRF = roll_test(tw_top_all, t50RF, t50cov_Rv, tw_mkt, optimize='MV', rolling_window=train_length, forcast_period=hold_period, 
                                          ml_macro=tw_macros, month_data=tw_month, nonN=True, shares_df=tw_top_shares, rf_df=tw_rf)

Et50_val_rvXG, Et50_wt_rvXG = roll_test(tw_top_all, t50XGB, t50cov_Rv, tw_mkt, optimize='MV', rolling_window=train_length, forcast_period=hold_period, 
                                          ml_macro=tw_macros, month_data=tw_month, nonN=True, shares_df=tw_top_shares, rf_df=tw_rf)
# Min CVaR
Ecvt50_val_rvRF, Ecvt50_wt_rvRF = roll_test(tw_top_all, t50RF, t50cov_Rv, tw_mkt, optimize='CVaR', rolling_window=train_length, forcast_period=hold_period, 
                                          ml_macro=tw_macros, month_data=tw_month, nonN=True, shares_df=tw_top_shares, rf_df=tw_rf)

Ecvt50_val_rvXG, Ecvt50_wt_rvXG = roll_test(tw_top_all, t50XGB, t50cov_Rv, tw_mkt, optimize='CVaR', rolling_window=train_length, forcast_period=hold_period, 
                                          ml_macro=tw_macros, month_data=tw_month, nonN=True, shares_df=tw_top_shares, rf_df=tw_rf)

# ML uncertainty
# Max Sharpe
Ut50_val_rvRF, Ut50_wt_rvRF = roll_test(tw_top_all, t50RF, t50cov_Rv, tw_mkt, optimize='MV', rolling_window=train_length, forcast_period=hold_period, 
                                          ml_macro=tw_macros, month_data=tw_month, uncertainty=t50RF_un, shares_df=tw_top_shares, rf_df=tw_rf)

Ut50_val_rvXG, Ut50_wt_rvXG = roll_test(tw_top_all, t50XGB, t50cov_Rv, tw_mkt, optimize='MV', rolling_window=train_length, forcast_period=hold_period, 
                                          ml_macro=tw_macros, month_data=tw_month, uncertainty=t50XGB_un, shares_df=tw_top_shares, rf_df=tw_rf)
# Min CVaR
Ucvt50_val_rvRF, Ucvt50_wt_rvRF = roll_test(tw_top_all, t50RF, t50cov_Rv, tw_mkt, optimize='CVaR', rolling_window=train_length, forcast_period=hold_period, 
                                          ml_macro=tw_macros, month_data=tw_month, uncertainty=t50RF_un, shares_df=tw_top_shares, rf_df=tw_rf)

Ucvt50_val_rvXG, Ucvt50_wt_rvXG = roll_test(tw_top_all, t50XGB, t50cov_Rv, tw_mkt, optimize='CVaR', rolling_window=train_length, forcast_period=hold_period, 
                                          ml_macro=tw_macros, month_data=tw_month, uncertainty=t50XGB_un, shares_df=tw_top_shares, rf_df=tw_rf)


# ML_uncertainty+Nonormal
#Min CVaR
UEcvt50_val_rvRF, UEcvt50_wt_rvRF = roll_test(tw_top_all, t50RF, t50cov_Rv, tw_mkt, optimize='CVaR', rolling_window=train_length, forcast_period=hold_period, 
                                          ml_macro=tw_macros, month_data=tw_month, nonN=True, uncertainty=t50RF_un, shares_df=tw_top_shares, rf_df=tw_rf)

UEcvt50_val_rvXG, UEcvt50_wt_rvXG = roll_test(tw_top_all, t50XGB, t50cov_Rv, tw_mkt, optimize='CVaR', rolling_window=train_length, forcast_period=hold_period, 
                                          ml_macro=tw_macros, month_data=tw_month, nonN=True, uncertainty=t50XGB_un, shares_df=tw_top_shares, rf_df=tw_rf)

# Benchmarks
mv_val, mv_wt = simple_roll_test(tw_top_all, 'MV', rolling_window=train_length, forcast_period=hold_period,  mkt=tw_mkt)

cvar_val, mv_wt = simple_roll_test(tw_top_all, 'CVaR', rolling_window=train_length, forcast_period=hold_period,  mkt=tw_mkt)

tw_eq_val_mv, tw_eq_wt_mv = simple_roll_test(tw_top_all, 'EQW', rolling_window=train_length, forcast_period=hold_period,  mkt=tw_mkt)

tw_mktCap_val_mv, tw_mktCap_wt_mv = simple_roll_test(tw_top_all, 'MKT', rolling_window=train_length, forcast_period=hold_period,  mkt=tw_mkt, share_df=tw_top_shares)


tw_mktCap_val_mv, tw_mktCap_wt_mv = roll_test(tw_top_all, 'XGB', 'cop_CV', tw_mkt, optimize='MV', rolling_window=train_length, forcast_period=hold_period, 
                                              ml_macro=tw_macros, month_data=tw_month, shares_df=tw_top_shares, rf_df=tw_rf, EQW='mkt')

tw_df_close = tw_top_all.loc[:,('Adj Close', slice(None))]
tw_df_close.columns = tw_df_close.columns.get_level_values(1)
tw_mkt_prices = pd.Series(tw_df_close[tw_mkt], name = tw_mkt)
tw_bench_start = (tw_top_all.index[0]+relativedelta(month=train_length)).strftime('%Y-%m')
tw_bench_end = tw_top_all.index[-1].strftime('%Y-%m')
tw_benchmark_prices = tw_mkt_prices[(tw_mkt_prices.index.strftime('%Y-%m') > tw_bench_start) & (tw_mkt_prices.index.strftime('%Y-%m') <= tw_bench_end)]
tw_benchmark_ret = tw_benchmark_prices.pct_change().dropna()
tw_benchmark_val = (tw_benchmark_prices/tw_benchmark_prices.iloc[0])

tw_benchmark_val.index = tw_mktCap_val_mv.index
tw_val_listDF = [
    t50_val_RF, t50_val_RFS, t50_val_rvRF, 
    t50_val_XG, t50_val_XGS, t50_val_rvXG, 
    cvt50_val_RF, cvt50_val_RFS, cvt50_val_rvRF,
    cvt50_val_XG, cvt50_val_XGS, cvt50_val_rvXG,
    Ecvt50_val_rvRF, Ecvt50_val_rvXG, 
    Ucvt50_val_rvRF, Ucvt50_val_rvXG,
    UEcvt50_val_rvRF, UEcvt50_val_rvXG,
    mv_val, cvar_val, tw_mktCap_val_mv, tw_eq_val_mv, [tw_benchmark_val]
    ]

tw_val_list = []
a = type(tw_mktCap_val_mv[0])
for i in tw_val_listDF:
    series = i[0]
    tw_val_list.append(series)

methods = [
    'Pearson_RF_MV', 'Shrink_RF_MV', 'Rvine_RF_MV', 
    'Pearson_XGB_MV', 'Shrink_XGB_MV', 'Rvine_XGB_MV',
    'Pearson_RF_CVaR', 'Shrink_RF_CVaR', 'Rvine_RF_CVaR',
    'Pearson_XGB_CVaR', 'Shrink_XGB_CVaR', 'Rvine_XGB_CVaR',
    'Non-Norm_Rvine_RF_CVaR', 
    'Non-Norm_Rvine_XGB_CVaR', 
    'ML_un_Rvine_RF_CVaR', 
    'ML_un_Rvine_XGB_CVaR', 
    'ML_un_Non-Norm_Rvine_RF_CVaR', 
    'ML_un_Non-Norm_Rvine_XGB_CVaR', 
    'maxSharpe', 'minCVaR', 'Market Cap', 'Equal_Weight', tw_mkt
    ]

tw_val_df = pd.DataFrame(data=dict(zip(methods, tw_val_list)))
tw_perf = performance_metric(tw_val_df, tw_rf)

############### plot ###############
best = ['Rvine_RF_MV', 'Rvine_XGB_MV','Rvine_RF_CVaR',
    'Pearson_XGB_CVaR', 'Rvine_XGB_CVaR',
    'ML_un_Non-Norm_Rvine_RF_CVaR', 'ML_un_Non-Norm_Rvine_XGB_CVaR', 
    'Equal_Weight', tw_mkt
    ]
copula = ['Pearson_RF_MV', 'Rvine_RF_MV', 
    'Pearson_XGB_MV','Rvine_XGB_MV',
    'Non-Norm_Rvine_RF_CVaR', 
    'Non-Norm_Rvine_XGB_CVaR', 
    'ML_un_Non-Norm_Rvine_RF_CVaR', 
    'ML_un_Non-Norm_Rvine_XGB_CVaR', 
   'Equal_Weight', tw_mkt
   ]
predictor = ['Pearson_RF_MV', 'Rvine_RF_MV', 
    'Pearson_XGB_MV', 'Rvine_XGB_MV',
    'Pearson_RF_CVaR', 'Rvine_RF_CVaR', 'Rvine_XGB_CVaR',
    'ML_un_Non-Norm_Rvine_RF_CVaR', 
    'ML_un_Non-Norm_Rvine_XGB_CVaR', 
    ]

title = ['Best Performers', 'Copula\'s Imporovements', 'Compare of Different Predictor']
for i, plot in enumerate([best, copula, predictor]):
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    for column in plot:
        plt.plot(tw_val_df.index, tw_val_df[column], label=column)

    plt.title('Portfolio Value - ' + title[i])
    plt.xlabel('Date')
    plt.ylabel('Values')

    plt.legend(ncol=2, fontsize='small', loc='upper left')
    plt.grid(axis='x', color='gray', alpha=0.15)
    plt.show()
    file_name = f'result/TW_results_{title[i]}.png'
    plt.savefig(file_name)

'''
tw_val_df.to_csv(f'results/{trade_period}_TW_TOP50_COP.csv', index=True)
tw_val_df.to_excel(f'results/{trade_period}_TW_TOP50_COP.xlsx', index=True)
tw_perf.to_excel(f'results/{trade_period}_TW_TOP50_PERF.xlsx')
'''

tw_wt_list = [
    t50_wt_RF, t50_wt_RFS, t50_wt_rvRF, 
    t50_wt_XG, t50_wt_XGS, t50_wt_rvXG, 
    cvt50_wt_RF, cvt50_wt_RFS, cvt50_wt_rvRF,
    cvt50_wt_XG, cvt50_wt_XGS, cvt50_wt_rvXG, 
    Ecvt50_wt_rvRF, 
    Ecvt50_wt_rvXG, 
    Ucvt50_wt_rvRF, 
    Ucvt50_wt_rvXG, 
    UEcvt50_wt_rvRF, 
    UEcvt50_wt_rvXG, 
    tw_eq_wt_mv, tw_mktCap_wt_mv
    ]

tw_wt_list[0].index.strftime('%Y-%M-%d')

"""
with pd.ExcelWriter(f'results/{trade_period}_TW_TOP50_COP_WTs.xlsx') as writer:
    # Iterate over each dataframe and method, and write each dataframe to a separate sheet
    for method, df in zip([x for x in methods if x != 'Benchmark'], tw_wt_list):
        df.to_excel(writer, sheet_name=method, index=True)
"""


################################################## US ############################################################
mkt = '^GSPC'
start='2019-01-01'
download_start = (pd.to_datetime(start) - relativedelta(months=1)).replace(day=1).strftime('%Y-%m-%d')
end = '2024-03-01'
pred_freq = '1d' #'1d', '1mo', '1y'
train_length = 12
hold_period = 1

top_all, top_shares, macros, month = us_data(mkt, 50, start, download_start, end, lag=20)
us_rf = pd.read_csv("data/US_risk_free_rate.csv", index_col=0)

#max Sharpe
u50_val_RF, u50_wt_RF = roll_test(top_all, u50RF, 'Pearson', mkt, optimize='MV', rolling_window=train_length, forcast_period=hold_period, 
                                  ml_macro=macros, month_data=month, rf_df=us_rf, shares_df=top_shares)
u50_val_RFS, u50_wt_RFS = roll_test(top_all, u50RF, 'Shrink', mkt, optimize='MV', rolling_window=train_length, forcast_period=hold_period, 
                                    ml_macro=macros, month_data=month, rf_df=us_rf, shares_df=top_shares)
u50_val_rvRF, u50_wt_rvRF = roll_test(top_all, u50RF, u50cov_Rv, mkt, optimize='MV', rolling_window=train_length, forcast_period=hold_period, 
                                      ml_macro=macros, month_data=month, rf_df=us_rf, shares_df=top_shares)

u50_val_XG, u50_wt_XG = roll_test(top_all, u50XGB, 'Pearson', mkt, optimize='MV', rolling_window=train_length, forcast_period=hold_period, 
                                    ml_macro=macros, month_data=month, rf_df=us_rf, shares_df=top_shares)
u50_val_XGS, u50_wt_XGS = roll_test(top_all, u50XGB, 'Shrink', mkt, optimize='MV', rolling_window=train_length, forcast_period=hold_period, 
                                      ml_macro=macros, month_data=month, rf_df=us_rf, shares_df=top_shares)
u50_val_rvXG, u50_wt_rvXG = roll_test(top_all, u50XGB, u50cov_Rv, mkt, rf_df=us_rf, optimize='MV', rolling_window=train_length, forcast_period=hold_period, 
                                        ml_macro=macros, month_data=month, shares_df=top_shares)

#min CVaR
cvu50_val_RF, cvu50_wt_RF = roll_test(top_all, u50RF, 'Pearson', mkt, optimize='CVaR', rolling_window=train_length, forcast_period=hold_period, 
                                        ml_macro=macros, month_data=month, rf_df=us_rf, shares_df=top_shares)
cvu50_val_RFS, cvu50_wt_RFS = roll_test(top_all, u50RF, 'Shrink', mkt, optimize='CVaR', rolling_window=train_length, forcast_period=hold_period, 
                                          ml_macro=macros, month_data=month, rf_df=us_rf, shares_df=top_shares)
cvu50_val_rvRF, cvu50_wt_rvRF = roll_test(top_all, u50RF, u50cov_Rv, mkt, optimize='CVaR', rolling_window=train_length, forcast_period=hold_period, 
                                            ml_macro=macros, month_data=month, rf_df=us_rf, shares_df=top_shares)

cvu50_val_XG, cvu50_wt_XG = roll_test(top_all, u50XGB, 'Pearson', mkt, optimize='CVaR', rolling_window=train_length, forcast_period=hold_period, 
                                        ml_macro=macros, month_data=month, rf_df=us_rf, shares_df=top_shares)
cvu50_val_XGS, cvu50_wt_XGS = roll_test(top_all, u50XGB, 'Shrink', mkt, optimize='CVaR', rolling_window=train_length, forcast_period=hold_period, 
                                          ml_macro=macros, month_data=month, rf_df=us_rf, shares_df=top_shares)
cvu50_val_rvXG, cvu50_wt_rvXG = roll_test(top_all, u50XGB, u50cov_Rv, mkt, optimize='CVaR', rolling_window=train_length, forcast_period=hold_period, 
                                            ml_macro=macros, month_data=month, rf_df=us_rf, shares_df=top_shares)

#non-normal
Ecvu50_val_RF, Ecvu50_wt_RF = roll_test(top_all, u50RF, 'Pearson', mkt, optimize='CVaR', rolling_window=train_length, forcast_period=hold_period, 
                                        ml_macro=macros, month_data=month, nonN=True, rf_df=us_rf, shares_df=top_shares)
Ecvu50_val_rvRF, Ecvu50_wt_rvRF = roll_test(top_all, u50RF, u50cov_Rv, mkt, optimize='CVaR', rolling_window=train_length, forcast_period=hold_period, 
                                            ml_macro=macros, month_data=month, nonN=True, rf_df=us_rf, shares_df=top_shares)

Ecvu50_val_XG, Ecvu50_wt_XG = roll_test(top_all, u50XGB, 'Pearson', mkt, optimize='CVaR', rolling_window=train_length, forcast_period=hold_period, 
                                        ml_macro=macros, month_data=month, nonN=True, rf_df=us_rf, shares_df=top_shares)
Ecvu50_val_rvXG, Ecvu50_wt_rvXG = roll_test(top_all, u50XGB, u50cov_Rv, mkt, optimize='CVaR', rolling_window=train_length, forcast_period=hold_period, 
                                            ml_macro=macros, month_data=month, nonN=True, rf_df=us_rf, shares_df=top_shares)


#uncertainty
Ucvu50_val_RF, Ucvu50_wt_RF = roll_test(top_all, u50RF, 'Pearson', mkt, optimize='CVaR', rolling_window=train_length, forcast_period=hold_period, 
                                        ml_macro=macros, month_data=month, uncertainty=u50RF_un, rf_df=us_rf, shares_df=top_shares)
Ucvu50_val_rvRF, Ucvu50_wt_rvRF = roll_test(top_all, u50RF, u50cov_Rv, mkt, optimize='CVaR', rolling_window=train_length, forcast_period=hold_period, 
                                            ml_macro=macros, month_data=month, uncertainty=u50RF_un, rf_df=us_rf, shares_df=top_shares)

Ucvu50_val_XG, Ucvu50_wt_XG = roll_test(top_all, u50XGB, 'Pearson', mkt, optimize='CVaR', rolling_window=train_length, forcast_period=hold_period, 
                                        ml_macro=macros, month_data=month, uncertainty=u50XGB_un, rf_df=us_rf, shares_df=top_shares)
Ucvu50_val_rvXG, Ucvu50_wt_rvXG = roll_test(top_all, u50XGB, u50cov_Rv, mkt, optimize='CVaR', rolling_window=train_length, forcast_period=hold_period, 
                                            ml_macro=macros, month_data=month, uncertainty=u50XGB_un, rf_df=us_rf, shares_df=top_shares)

#uncertainty+non-normal
UEcvu50_val_RF, UEcvu50_wt_RF = roll_test(top_all, u50RF, 'Pearson', mkt, optimize='CVaR', rolling_window=train_length, forcast_period=hold_period, 
                                        ml_macro=macros, month_data=month, nonN=True, uncertainty=u50RF_un, rf_df=us_rf, shares_df=top_shares)
UEcvu50_val_rvRF, UEcvu50_wt_rvRF = roll_test(top_all, u50RF, u50cov_Rv, mkt, optimize='CVaR', rolling_window=train_length, forcast_period=hold_period, 
                                            ml_macro=macros, month_data=month, nonN=True, uncertainty=u50RF_un, rf_df=us_rf, shares_df=top_shares)

UEcvu50_val_XG, UEcvu50_wt_XG = roll_test(top_all, u50XGB, 'Pearson', mkt, optimize='CVaR', rolling_window=train_length, forcast_period=hold_period, 
                                        ml_macro=macros, month_data=month, nonN=True, uncertainty=u50XGB_un, rf_df=us_rf, shares_df=top_shares)
UEcvu50_val_rvXG, UEcvu50_wt_rvXG = roll_test(top_all, u50XGB, u50cov_Rv, mkt, optimize='CVaR', rolling_window=train_length, forcast_period=hold_period, 
                                            ml_macro=macros, month_data=month, nonN=True, uncertainty=u50XGB_un, rf_df=us_rf, shares_df=top_shares)

# Benchmark
us_mv_val, us_mv_wt = simple_roll_test(top_all, 'MV', rolling_window=train_length, forcast_period=hold_period, mkt=mkt)

us_cvar_val, us_cvar_wt = simple_roll_test(top_all, 'CVaR', rolling_window=train_length, forcast_period=hold_period, mkt=mkt)

us_eq_val_mv, us_eq_wt_mv = simple_roll_test(top_all, 'EQW', rolling_window=train_length, forcast_period=hold_period, mkt=mkt)

us_mktCap_val_mv, us_mktCap_wt_mv = simple_roll_test(top_all, 'MKT', rolling_window=train_length, forcast_period=hold_period, mkt=mkt, share_df=top_shares)


# Store Portfolios' Value
df_close = top_all.loc[:,('Adj Close', slice(None))]
df_close.columns = df_close.columns.get_level_values(1)
mkt_prices = pd.Series(df_close[mkt], name = mkt)
bench_start = (top_all.index[0]+relativedelta(month=train_length)).strftime('%Y-%m')
bench_end = top_all.index[-1].strftime('%Y-%m')
benchmark_prices = mkt_prices[(mkt_prices.index.strftime('%Y-%m') > bench_start) & (mkt_prices.index.strftime('%Y-%m') <= bench_end)]
benchmark_ret = benchmark_prices.pct_change().dropna()
benchmark_val = (benchmark_prices/benchmark_prices.iloc[0])

benchmark_val.index = us_eq_val_mv.index

val_listDF = [
    u50_val_RF, u50_val_RFS, u50_val_rvRF,
    u50_val_XG, u50_val_XGS, u50_val_rvXG, 
    cvu50_val_RF, cvu50_val_RFS, cvu50_val_rvRF, 
    cvu50_val_XG, cvu50_val_XGS, cvu50_val_rvXG, 
    Ecvu50_val_rvRF, 
    Ecvu50_val_rvXG, 
    Ucvu50_val_rvRF,
    Ucvu50_val_rvXG, 
    UEcvu50_val_rvRF, 
    UEcvu50_val_rvXG,
    mv_val, cvar_val, us_mktCap_val_mv, us_eq_val_mv, [benchmark_val]
    ]
val_list = []
for i in val_listDF:
    print(i)
    series = i[0]
    val_list.append(series)

methods = [
    'Pearson_RF_MV', 'Shrink_RF_MV', 'Rvine_RF_MV', 
    'Pearson_XGB_MV', 'Shrink_XGB_MV', 'Rvine_XGB_MV',
    'Pearson_RF_CVaR', 'Shrink_RF_CVaR', 'Rvine_RF_CVaR',
    'Pearson_XGB_CVaR', 'Shrink_XGB_CVaR', 'Rvine_XGB_CVaR',
    'Non-Norm_Rvine_RF_CVaR', 
    'Non-Norm_Rvine_XGB_CVaR', 
    'ML_un_Rvine_RF_CVaR', 
    'ML_un_Rvine_XGB_CVaR', 
    'ML_un_Non-Norm_Rvine_RF_CVaR', 
    'ML_un_Non-Norm_Rvine_XGB_CVaR', 
    'maxSharpe', 'minCVaR', 'Market Cap', 'Equal_Weight', mkt
    ]


val_df = pd.DataFrame(data=dict(zip(methods, val_list)))
us_perf = performance_metric(val_df, us_rf)

############### plot ###############
best = ['Rvine_RF_MV', 'Rvine_XGB_MV','Rvine_RF_CVaR',
    'Pearson_XGB_CVaR', 'Rvine_XGB_CVaR',
    'ML_un_Non-Norm_Rvine_RF_CVaR', 'ML_un_Non-Norm_Rvine_XGB_CVaR', 
    'Equal_Weight', mkt
    ]
copula = ['Pearson_RF_MV', 'Rvine_RF_MV', 
    'Pearson_XGB_MV','Rvine_XGB_MV',
    'Non-Norm_Rvine_RF_CVaR', 
    'Non-Norm_Rvine_XGB_CVaR', 
    'ML_un_Non-Norm_Rvine_RF_CVaR', 
    'ML_un_Non-Norm_Rvine_XGB_CVaR', 
   'Equal_Weight', mkt
   ]
predictor = ['Pearson_RF_MV', 'Rvine_RF_MV', 
    'Pearson_XGB_MV', 'Rvine_XGB_MV',
    'Pearson_RF_CVaR', 'Rvine_RF_CVaR', 'Rvine_XGB_CVaR',
    'ML_un_Non-Norm_Rvine_RF_CVaR', 
    'ML_un_Non-Norm_Rvine_XGB_CVaR', 
    ]

title = ['Best Performers', 'Copula\'s Imporovements', 'Compare of Different Predictor']
for i, plot in enumerate([best, copula, predictor]):
    plt.figure(figsize=(10, 6))  # Adjust the figure size as needed
    for column in plot:
        plt.plot(val_df.index, val_df[column], label=column)
    plt.title('Portfolio Value - ' + title[i])
    plt.xlabel('Date')
    plt.ylabel('Values')

    plt.legend(ncol=2, fontsize='small', loc='upper left')
    plt.grid(axis='x', color='gray', alpha=0.15)
    #plt.show()
    file_name = f'result/US_results_{title[i]}.png'
    plt.savefig(file_name)

val_df.index
tw_val_df.index

'''
val_df.to_csv(f'results/{trade_period}_US_TOP50_COP.csv', index=True)
val_df.to_excel(f'results/{trade_period}_US_TOP50_COP.xlsx', index=True)
us_perf.to_excel(f'results/{trade_period}_US_TOP50_PERF.xlsx', index=True)
'''

# Save weight changes
wt_list = [
    u50_wt_RF, u50_wt_RFS, u50_wt_rvRF,
    u50_wt_XG, u50_wt_XGS, u50_wt_rvXG,
    cvu50_wt_RF, cvu50_wt_RFS, cvu50_wt_rvRF,
    cvu50_wt_XG, cvu50_wt_XGS, cvu50_wt_rvXG, 
    Ecvu50_wt_rvRF, 
    Ecvu50_wt_rvXG,
    Ucvu50_wt_rvRF, 
    Ucvu50_wt_rvXG, 
    UEcvu50_wt_rvRF, 
    UEcvu50_wt_rvXG,
    us_eq_wt_mv, us_mktCap_wt_mv
    ]

wt_list[0].index.strftime('%Y-%M-%d')
with pd.ExcelWriter(f'results/{trade_period}_US_TOP50_COP_WTs.xlsx') as writer:
    # Iterate over each dataframe and method, and write each dataframe to a separate sheet
    for method, df in zip([x for x in methods if x != 'Benchmark'], wt_list):
        df.to_excel(writer, sheet_name=method, index=True)
