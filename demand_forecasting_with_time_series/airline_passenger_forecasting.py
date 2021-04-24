#################################
# Airline Passenger Forecasting
#################################


import itertools
import warnings
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import statsmodels.api as sm
import statsmodels.tsa.api as smt
from statsmodels.tsa.arima_model import ARIMA
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error
warnings.filterwarnings('ignore')


#################################
# Visualizing Data
#################################

df = pd.read_csv('datasets/airline-passengers.csv', index_col='month',
                 parse_dates=True)
df.shape
df.head()

df[['total_passengers']].plot(title='Passengers Data')
plt.show()

# Let's express that index will be monthly.
df.index.freq = "MS"

# I choose 120 observations as a test, and train 24 observations.
train = df[:120]
test = df[120:]


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
    ts_decompose(df[['total_passengers']], model, True)


#################################
# Single Exponential Smoothing
#################################
# smoothing_level = alpha parameter = learning coefficient -> level
def optimize_ses(train, alphas, step=48):
    for alpha in alphas:
        ses_model = SimpleExpSmoothing(train).fit(smoothing_level=alpha)
        y_pred = ses_model.forecast(step)
        mae = mean_absolute_error(test, y_pred)
        print("alpha:", round(alpha, 2), "mae:", round(mae, 4))


alphas = np.arange(0.01, 1, 0.10)
# step 24 because test size is 24 months, test.shape = (24, 1)
optimize_ses(train, alphas, step=24)
# alpha: 0.11 mae: 82.528

ses_model = SimpleExpSmoothing(train).fit(smoothing_level=0.11)
y_pred = ses_model.forecast(24)


def plot_prediction(y_pred, label):
    train["total_passengers"].plot(legend=True, label="TRAIN")
    test["total_passengers"].plot(legend=True, label="TEST")
    y_pred.plot(legend=True, label="PREDICTION")
    plt.title("Train, Test and Predicted Test Using "+label)
    plt.show()


plot_prediction(y_pred, "Single Exponential Smoothing")


#################################
# Double Exponential Smoothing
#################################
# smoothing_level = alpha parameter = learning coefficient -> level
# smoothing_slope = beta parameter = remembering coefficient -> trend
def optimize_des(train, alphas, betas, step=48):
    print("Optimizing parameters...")
    results = []
    for alpha in alphas:
        for beta in betas:
            des_model = ExponentialSmoothing(train, trend="add").\
                fit(smoothing_level=alpha, smoothing_slope=beta)
            y_pred = des_model.forecast(step)
            mae = mean_absolute_error(test, y_pred)
            results.append([round(alpha, 2), round(beta, 2), round(mae, 2)])
    results = pd.DataFrame(results, columns=["alpha", "beta", "mae"]).\
        sort_values("mae")
    print(results)


alphas = np.arange(0.01, 1, 0.10)
betas = np.arange(0.01, 1, 0.10)
optimize_des(train, alphas, betas, step=24)
# 54.33

des_model = ExponentialSmoothing(train, trend="add").fit(smoothing_level=0.01,
                                                         smoothing_slope=0.11)
y_pred = des_model.forecast(24)

plot_prediction(y_pred, "Double Exponential Smoothing")


#################################
# Triple Exponential Smoothing (Holt-Winters)
#################################
# smoothing_level = alpha parameter = learning coefficient -> level
# smoothing_slope = beta parameter = remembering coefficient -> trend
# smoothing_seasonal = gama parameter -> seasonality coefficient ->
# seasonality

def optimize_tes(train, abg, step=48):
    print("Optimizing parameters...")
    results = []
    for comb in abg:
        tes_model = ExponentialSmoothing(train, trend="add", seasonal="add",
                                         seasonal_periods=12).\
            fit(smoothing_level=comb[0], smoothing_slope=comb[1],
                smoothing_seasonal=comb[2])
        y_pred = tes_model.forecast(step)
        mae = mean_absolute_error(test, y_pred)
        print([round(comb[0], 2), round(comb[1], 2), round(comb[2], 2),
               round(mae, 2)])
        results.append([round(comb[0], 2), round(comb[1], 2), round(comb[2], 2),
                        round(mae, 2)])
    results = pd.DataFrame(results, columns=["alpha", "beta", "gamma", "mae"]).\
        sort_values("mae")
    print(results)

alphas = betas = gammas = np.arange(0.10, 1, 0.20)
abg = list(itertools.product(alphas, betas, gammas))

optimize_tes(train, abg, step=24)
# 11.99
# multiplicative error: 15.12

tes_model = ExponentialSmoothing(train, trend="add", seasonal="add",
                                 seasonal_periods=12).\
            fit(smoothing_level=0.3, smoothing_slope=0.3,
                smoothing_seasonal=0.5)

y_pred = tes_model.forecast(24)

plot_prediction(y_pred, "Triple Exponential Smoothing ADD")


##################################################
# ARIMA(p, d, q): (Autoregressive Integrated Moving Average)
##################################################

# Generating combinations of p and q
p = d = q = range(0, 4)
pdq = list(itertools.product(p, d, q))

# the smaller the aic the better, we are concerned about the smaller the aic


def arima_optimizer_aic(train, orders):
    best_aic, best_params = float("inf"), None
    for order in orders:
        # ARIMA may not be able to run all combinations within orders.
        # When we get an error, we run it within try, except to continue
        # working. In the except part, we write continue, so that after the
        # non-working value, the next value immediately goes.
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


# Tuned Model
arima_model = ARIMA(train, best_params_aic).fit(disp=0)

# arima_model.summary()
y_pred = arima_model.forecast(24)[0]
# arima_model.forecast(24)[0]: forecasted values
# arima_model.forecast(24)[1]: forecast errors
# arima_model.forecast(24)[2]: confidence intervals
mean_absolute_error(test, y_pred)
# 51.18

plot_prediction(pd.Series(y_pred, index=test.index), "ARIMA")

# !!! We do not expect to get better results as we progress in these models, we
# try different methods and different models for our prediction and we see their
# errors.

##################################################
# SARIMA
##################################################

p = d = q = range(0, 2)
pdq = list(itertools.product(p, d, q))
seasonal_pdq = [(x[0], x[1], x[2], 12) for x in list(itertools.product(p, d, q))]


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

# Tuned Model
model = SARIMAX(train, order=best_order, seasonal_order=best_seasonal_order)
sarima_final_model = model.fit(disp=0)
y_pred_test = sarima_final_model.get_forecast(steps=24)
pred_ci = y_pred_test.conf_int()
y_pred = y_pred_test.predicted_mean
mean_absolute_error(test, y_pred)
# 54.66

plot_prediction(pd.Series(y_pred, index=test.index), "SARIMA")