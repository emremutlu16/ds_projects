##################################################
# SARIMA(p, d, q): (Seasonal Autoregressive Integrated Moving-Average)
##################################################

import itertools
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.tsa.api as smt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

pd.set_option('display.float_format', lambda x: '%0.2' % x)
np.set_printoptions(suppress=True, formatter={'float_kind': '{:0.2f}'.format})
warnings.filterwarnings('ignore')

##################################################
# Data Set
##################################################

data = sm.datasets.co2.load_pandas()
y = data.data
y = y['co2'].resample('MS').mean()
y = y.fillna(y.bfill())

y.plot(figsize=(15, 6))
plt.show()


train = y[:'1997-12-01']
test = y['1998-01-01':]
val = train['1994-01-01':"1997-12-01"]


#################################
# Structural Analysis of Time Series
#################################
def ts_decompose(y, model="additive", stationary=False):
    result = seasonal_decompose(y, model=model)
    fig, axes = plt.subplots(4, 1, sharex=True, sharey=False)
    fig.set_figheight(10)
    fig.set_figwidth(15)

    axes[0].set_title("Decomposition for " + model + " model")
    axes[0].plot(y, 'k', label='Original ' + model)
    axes[0].legend(loc='upper left')

    axes[1].plot(result.trend, label='Trend')
    axes[1].legend(loc='upper left')

    axes[2].plot(result.seasonal, 'g',
                 label='Seasonality & Mean: ' +
                       str(round(result.seasonal.mean(), 4)))
    axes[2].legend(loc='upper left')

    axes[3].plot(result.resid, 'r',
                 label='Residuals & Mean: ' +
                       str(round(result.resid.mean(), 4)))
    axes[3].legend(loc='upper left')
    plt.show()

    if stationary:
        print("HO: Series is not Stationary.")
        print("H1: Series is Stationary.")
        p_value = sm.tsa.stattools.adfuller(y)[1]
        if p_value < 0.05:
            print(F"Result: Series is Stationary ({p_value}).")
        else:
            print(F"Result: Series is not Stationary ({p_value}).")

for model in ["additive", "multiplicative"]:
    ts_decompose(y, model, True)

##################################################
# MODEL
##################################################

model = SARIMAX(train, order=(1, 0, 1), seasonal_order=(0, 0, 0, 12))
sarima_model = model.fit(disp=0)

##################################################
# Validation Error
##################################################

pred = sarima_model.get_prediction(start=pd.to_datetime('1994-01-01'),
                                   dynamic=False)
pred_ci = pred.conf_int()

y_pred = pred.predicted_mean
mean_absolute_error(val, y_pred)
# 0.80

##################################################
# Visualizing Validation Estimates
##################################################

ax = train["1985":].plot(label='TRAIN')
pred.predicted_mean.plot(ax=ax, label='VALIDATION FORECASTS', alpha=.7)
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)

ax.set_xlabel('Date')
ax.set_ylabel('CO2 Levels')
plt.legend()
plt.show()

##################################################
# Test Error
##################################################
y_pred_test = sarima_model.get_forecast(steps=48)
pred_ci = y_pred_test.conf_int()
mean_absolute_error(test, y_pred)
# 11.48


##################################################
# Visualizing Test Estimates
##################################################


ax = y["1985":].plot(label='TRAIN')
test.plot(legend=True, label="TEST")
y_pred_test.predicted_mean.plot(ax=ax, label='TEST FORECASTS', alpha=.7)
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)

ax.set_xlabel('Date')
ax.set_ylabel('CO2 Levels')
plt.legend()
plt.title("Forecast vs Real for Test")
plt.show()

##################################################
# MODEL Tuning
##################################################

p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in
                list(itertools.product(p, d, q))]


def sarima_optimizer_aic(train, pdq, seasonal_pdq):
    best_aic, best_order, best_seasonal_order = float("inf"), float("inf"), None
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                sarimax_model = SARIMAX(train, order=param,
                                        seasonal_order=param_seasonal)
                results = sarimax_model.fit(disp=0)
                aic = results.aic
                if aic < best_aic:
                    best_aic, best_order, best_seasonal_order = aic, param,\
                                                                param_seasonal
                print('SARIMA{}x{}12 - AIC:{}'.format(param, param_seasonal,
                                                      aic))
            except:
                continue
    print('SARIMA{}x{}12 - AIC:{}'.format(best_order, best_seasonal_order,
                                          best_aic))
    return best_order, best_seasonal_order


best_order, best_seasonal_order = sarima_optimizer_aic(train, pdq, seasonal_pdq)

##################################################
# Final Model
##################################################

model = SARIMAX(train, order=best_order, seasonal_order=best_seasonal_order)
sarima_final_model = model.fit(disp=0)


##################################################
# Final Model Test Error
##################################################

y_pred_test = sarima_final_model.get_forecast(steps=48)
pred_ci = y_pred_test.conf_int()
mean_absolute_error(test, y_pred)

##################################################
# Visualization of Final Model Test Predictions
##################################################

ax = y["1985":].plot(label='TRAIN')
test.plot(legend=True, label="TEST")
y_pred_test.predicted_mean.plot(ax=ax, label='TEST FORECASTS', alpha=.7)
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)

ax.set_xlabel('Date')
ax.set_ylabel('CO2 Levels')
plt.legend()
plt.show()

##################################################
# Analyzing the Statistical Outputs of the Model
##################################################

sarima_final_model.plot_diagnostics(figsize=(15, 12))
plt.show()


##################################################
# SARIMA Optimizer According to MAE
##################################################


def fit_model_sarima(train, val, pdq, seasonal_pdq):
    sarima_model = SARIMAX(train, order=pdq, seasonal_order=seasonal_pdq).\
        fit(disp=0)
    y_pred_val = sarima_model.get_forecast(steps=48)
    y_pred = y_pred_val.predicted_mean
    return mean_absolute_error(val, y_pred)

fit_model_sarima(train, val, (0, 1, 0), (0, 0, 0, 12))

p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in
                list(itertools.product(p, d, q))]


def sarima_optimizer_mae(train, val, pdq, seasonal_pdq):
    best_mae, best_order, best_seasonal_order = float("inf"), float("inf"), None
    for param in pdq:
        for param_seasonal in seasonal_pdq:
            try:
                mae = fit_model_sarima(train, val, param, param_seasonal)
                if mae < best_mae:
                    best_mae, best_order, best_seasonal_order = mae, param,\
                                                                param_seasonal
                print('SARIMA{}x{}12 - MAE:{}'.format(param, param_seasonal,
                                                      mae))
            except:
                continue
    print('SARIMA{}x{}12 - MAE:{}'.format(best_order, best_seasonal_order,
                                          best_mae))
    return best_order, best_seasonal_order


best_order, best_seasonal_order = sarima_optimizer_mae(train, val, pdq,
                                                       seasonal_pdq)


model = SARIMAX(train, order=best_order, seasonal_order=best_seasonal_order)
sarima_final_model = model.fit(disp=0)

y_pred_test = sarima_final_model.get_forecast(steps=48)
pred_ci = y_pred_test.conf_int()
y_pred = y_pred_test.predicted_mean
mean_absolute_error(test, y_pred)
# 1.024


ax = y["1985":].plot(label='TRAIN')
test.plot(legend=True, label="TEST")
y_pred_test.predicted_mean.plot(ax=ax, label='TEST FORECASTS', alpha=.7)
ax.fill_between(pred_ci.index,
                pred_ci.iloc[:, 0],
                pred_ci.iloc[:, 1], color='k', alpha=.2)

ax.set_xlabel('Date')
ax.set_ylabel('CO2 Levels')
plt.legend()
plt.show()
