from math import *
import itertools
from time import time
import pandas as pd
import numpy as np

import arch
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import (plot_acf,
                                           plot_pacf)

from tabulate import tabulate


def model(ts, name, log_values=False, train=0.95, rescale=100, arima=(0, 0, 0), garch=(1, 0, 0), auto_solve=False):
    """
    Select the best time-series model.
    :param log_values: Transform values into logarithmic notation.
    :param auto_solve: 
    :param rescale: 
    :param garch: General Auto-Regressive Conditionalm Heterokedasticity - Used to model volatility.
    :param arima: Auto-Regressive Integrated Moving Average
    :param ts: Time Series dataset
    :param name: Name to use
    :param train:
    :return:
    """
    start = time()

    if isinstance(ts, pd.DataFrame) and len(ts.columns) >= 2:
        return print(f'As of today, only one series can be applied.'
                     f'Please, re-run the function with only one series.')

    if log_values:
        ts = np.log(ts)  # Apply Natural Log to the series.

    # Data parsing
    n = len(ts)
    mask = floor(n*train)
    ts = ts.asfreq(pd.infer_freq(ts.index))
    ts_train, ts_test = ts.iloc[:mask+1], ts.iloc[mask:]
    if len(set(ts_train.index.year)) > 5:
        lookback = ts_test.index[0] - pd.offsets.DateOffset(years=5)
    else:
        lookback = ts_train.index[0]

    # Model Construction and Fit
    # Representation of Orders

    if auto_solve:
        rmse_mask = dict(arima=None, garch=None, RMSE=1e4)

        max_orders_arima = [np.arange(5), np.arange(3), np.arange(5)]
        max_orders_garch = [np.arange(start=1, stop=3), np.arange(2), np.arange(3)]
        arima_permutations = list(itertools.product(*max_orders_arima))
        garch_permutations = list(itertools.product(*max_orders_garch))

        for order in arima_permutations:

            # ARIMA
            train = sm.tsa.arima.ARIMA(ts_train * rescale,
                                       order=order,
                                       enforce_stationarity=False,
                                       enforce_invertibility=False, )
            results = train.fit()

            for garch_order in garch_permutations:

                garch_train = arch.arch_model(results.resid.dropna(), p=int(garch_order[0]),
                                              o=int(garch_order[1]), q=int(garch_order[2]))
                garch_results = garch_train.fit(disp='off')

                try:
                    predictions = results.get_prediction(start=ts_test.index[0],
                                                         end=ts_test.index[-1],
                                                         dynamic=True,
                                                         full_results=True)
                except KeyError:
                    predictions = results.get_prediction(start=1,
                                                         end=len(ts_test),
                                                         dynamic=True,
                                                         full_results=True)

                # GARCH
                garch_predictions = garch_results.forecast(horizon=len(ts_test))
                if garch_results.params.get('omega'):
                    garch_forecast = garch_results.params.get('omega') * np.sqrt(
                        garch_predictions.mean.dropna().values[0])
                else:
                    garch_forecast = np.sqrt(garch_predictions.mean.dropna().values[0])

            # Forecast and Risk Functions
                forecast = np.exp((predictions.predicted_mean + garch_forecast) * rescale**-1)
                rmse = np.sqrt(((forecast.values - np.exp(ts_test.values))**2).mean())

                if isnan(float(rmse)):
                    continue

                print(tabulate([
                    ['ARIMA Order', order],
                    ['GARCH Order', garch_order],
                    ['RMSE', rmse]
                ], headers=['Name', 'Order']))

                if float(rmse) < rmse_mask.get('RMSE'):
                    rmse_mask.update({'arima': order, 'garch': garch_order, 'RMSE': rmse})

        order = rmse_mask.get('arima')
        garch_order = rmse_mask.get('garch')
    else:
        order = arima
        garch_order = garch

    # Models and summary statistics
    train = sm.tsa.arima.ARIMA(ts_train*rescale,
                               order=order,
                               enforce_stationarity=False,
                               enforce_invertibility=False,)
    results = train.fit()
    print(results.summary())

    garch_train = arch.arch_model(results.resid.dropna(), p=int(garch_order[0]),
                                  o=int(garch_order[1]), q=(garch_order[2]))
    garch_results = garch_train.fit(disp='off')
    print(garch_results.summary())

    # Elapsed Time
    end = time()
    print(f'Process took approximately {round(end-start, 2)} seconds.')

    # Predictions
    # ARIMA
    try:
        predictions = results.get_prediction(start=ts_test.index[0],
                                             end=ts_test.index[-1],
                                             dynamic=True,
                                             full_results=True)
    except KeyError:
        predictions = results.get_prediction(start=1,
                                             end=len(ts_test),
                                             dynamic=True,
                                             full_results=True)

    pred_ci = np.exp(predictions.conf_int() * rescale**-1)

    # GARCH
    garch_predictions = garch_results.forecast(horizon=len(ts_test))
    if garch_results.params.get('omega'):
        garch_forecast = garch_results.params.get('omega')*np.sqrt(garch_predictions.mean.dropna().values[0])
    else:
        garch_forecast = np.sqrt(garch_predictions.mean.dropna().values[0])

    # Error estimation
    error = garch_results.params.get('omega')*np.sqrt(garch_predictions.mean.dropna().values[0] +
                                                      (garch_results.params.get('alpha[1]') *
                                                       garch_results.resid.values[-1]**2))

    # Forecast and Risk Functions
    forecast = np.exp((predictions.predicted_mean + garch_forecast) * rescale**-1)
    forecast_2 = np.exp((predictions.predicted_mean + error) * rescale**-1)
    forecast.index = pred_ci.index = forecast_2.index = ts_test.index  # Make sure that all indexes are equal
    mse = ((forecast.values - np.exp(ts_test.values)) ** 2).mean()
    mae = np.mean(np.abs(np.exp(ts_test.values)-forecast.values))

    # Plot the data
    fig = plt.figure(figsize=(12, 8), num=f'ARIMA {str(order)} + GARCH {str(garch_order)} for {name}')
    layout = (2, 2)

    fig.suptitle(f'Training set from {ts_train.index[0].year} to {ts_train.index[-1].year}\n'
                 f'Forecast begins {ts_test.index[0].strftime("%Y-%m")} and '
                 f'ends on {ts_test.index[-1].strftime("%Y-%m")}\n',
                 fontsize=14)

    # Aggregate Data
    ax = plt.subplot2grid(layout, (0, 0), colspan=2)
    acf_axis = plt.subplot2grid(layout, (1, 0))
    pacf_axis = plt.subplot2grid(layout, (1, 1))

    (np.exp(ts_train)).loc[lookback:].squeeze().plot(label=f'Training Data', ax=ax)
    (np.exp(ts_test)).squeeze().plot(label=f'Testing Data', ax=ax)
    forecast.plot(ax=ax, label=f'One-Step Ahead Forecast', alpha=0.5)
    forecast_2.plot(ax=ax, label=f'ARCH Error Estimation', alpha=0.2)

    # Residuals
    plot_acf(results.resid.dropna().values+garch_results.resid.values, ax=acf_axis)
    plot_pacf(results.resid.dropna().values+garch_results.resid.values, ax=pacf_axis)

    # Specs
    # Back-Testing
    [ax.axvline(f'{year}-01-01',
                color='k', linestyle='--', alpha=0.2) for year in set(ts_train.index.year)
     if pd.to_datetime(f'{year}-01-01', format=f'%Y-%m-%d') >= ts_train.index[0]]

    ax.fill_between(pred_ci.index,
                    pred_ci.iloc[:, 0],
                    pred_ci.iloc[:, 1], color='k', alpha=0.2)
    ax.fill_betweenx(ax.get_ylim(), ts_test.index[0], ts_test.index[-1],
                     alpha=.1, zorder=-1)
    ax.set(xlabel=f'Date', ylabel=f'Value',)
    ax.set_title(label=f"The Root Mean-Square Error (RMSE) of our forecast is {round(np.sqrt(mse), 2):,}\n"
                       f"The Mean Absolute Error (MAE) of our forecast is {round(float(mae), 5)}",
                 fontweight='bold')
    ax.legend(loc='best')

    [ax.set_xlim(0) for ax in [acf_axis, pacf_axis]]
    [sns.despine(ax=ax) for ax in [acf_axis, pacf_axis]]

    plt.tight_layout()
    plt.show()

    adf = sm.tsa.stattools.adfuller(results.resid.values)

    print(f'\nADF Test Statistic for residuals is: {round(adf[0], 2)}\n'
          f'p-value: {round(adf[1], 2)}\n'
          f'Critical Values:')
    for key, value in adf[4].items():
        print(f'\t{key}: {round(value, 2)}')
    print(f'\n')

    print(tabulate([
        ['One-Step Ahead Forecast', round(np.sqrt(mse), 2),
          round(float(mae), 5)],
        ['ARCH Error Forecast',
         round(np.sqrt(((forecast_2.values - np.exp(ts_test.values)) ** 2).mean()), 2),
         round(float(np.mean(np.abs(np.exp(ts_test.values)-forecast_2.values))), 5)]
        ], headers=['Name', 'RMSE', 'MAE']))
