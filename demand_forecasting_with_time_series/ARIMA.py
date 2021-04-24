##################################################
# ARIMA(p, d, q): (Autoregressive Integrated Moving Average)
##################################################

# Used in trend series. It will not work well if there is seasonality.

import itertools
import warnings
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.tsa.api as smt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose

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
arima_model = ARIMA(train, order=(1, 1, 1)).fit(disp=0)  # order(p, d, q)
arima_model.summary()


y_pred = arima_model.forecast(48)[0]
mean_absolute_error(test, y_pred)
# 2.7193

arima_model.plot_predict(dynamic=False)
plt.show()

train["1985":].plot(legend=True, label="TRAIN")
test.plot(legend=True, label="TEST", figsize=(6, 4))
pd.Series(y_pred, index=test.index).plot(legend=True, label="PREDICTION")
plt.title("Train, Test and Predicted Test")
plt.show()


##################################################
# MODEL TUNING
##################################################

##################################################
# Statistical Consideration of Model Degree Selection
##################################################

# 1. Determining the Model Rank According to ACF & PACF Graphs
# 2. Determining the Model Rank According to AIC Statistics
######################
# Determining the Model Rank According to ACF & PACF Graphs
######################

def acf_pacf(y, lags=30):
    plt.figure(figsize=(12, 7))
    layout = (2, 2)
    ts_ax = plt.subplot2grid(layout, (0, 0), colspan=2)
    acf_ax = plt.subplot2grid(layout, (1, 0))
    pacf_ax = plt.subplot2grid(layout, (1, 1))
    y.plot(ax=ts_ax)

    # Stationary test (HO: Series is not Stationary. H1: Series is Stationary.)
    p_value = sm.tsa.stattools.adfuller(y)[1]
    ts_ax.set_title('Time Series Analysis Plots\n Dickey-Fuller: p={0:.5f}'.
                    format(p_value))
    smt.graphics.plot_acf(y, lags=lags, ax=acf_ax)
    smt.graphics.plot_pacf(y, lags=lags, ax=pacf_ax)
    plt.tight_layout()
    plt.show()


acf_pacf(y)


df_diff = y.diff()
df_diff.dropna(inplace=True)

acf_pacf(df_diff)

##################################################
# Determining the Model Rank According to AIC & BIC Statistics
##################################################

# Generating combinations of p and q
p = d = q = range(0, 4)
pdq = list(itertools.product(p, d, q))

# ARIMA(1,1,1)


def arima_optimizer_aic(train, orders):
    best_aic, best_params = float("inf"), None

    for order in orders:
        try:
            arma_model_result = ARIMA(train, order).fit(disp=0)
            aic = arma_model_result.aic
            if aic < best_aic:
                best_aic, best_params = aic, order
            print('ARIMA%s AIC=%.2f' % (order, aic))
        except:
            continue
    print('Best ARIMA%s AIC=%.2f' % (best_params, best_aic))
    return best_params


best_params_aic = arima_optimizer_aic(train, pdq)

##################################################
# Tuned Model
##################################################

arima_model = ARIMA(train, best_params_aic).fit(disp=0)
y_pred = arima_model.forecast(48)[0]
mean_absolute_error(test, y_pred)
# 1.55

train["1985":].plot(legend=True, label="TRAIN")
test.plot(legend=True, label="TEST", figsize=(6, 4))
pd.Series(y_pred, index=test.index).plot(legend=True, label="PREDICTION")
plt.title("Train, Test and Predicted Test")
plt.show()
