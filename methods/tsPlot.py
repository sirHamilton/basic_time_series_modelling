"""
Visualize Time Series Data
Construct and Fit Models
Analyze the Findings
Inference and Remarks
"""

# Import necessary packages
import os
import requests

from fredapi import Fred

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from statsmodels.graphics.tsaplots import (plot_acf,
                                           plot_pacf)


def economics_series(series, end_date=None, difference=None, log=False):
    fred = Fred(api_key=os.environ.get('FRED_API'))  # Instantiate - Use for economic data (i.e. get_series())
    df = fred.get_series(str(series))
    if end_date and isinstance(end_date, str):
        df = df.loc[:end_date]
    df = df.asfreq(pd.infer_freq(df.index))
    if difference and isinstance(difference, int):
        diff = ".diff()" * difference
        df = eval('np.log(df)'+diff+'.dropna()')
        return df
    if log:
        return np.log(df)
    return df


def equity_lookup(equity_list, difference=None):

    collection = dict()

    configuration = dict(
        frequencyType='daily',
        frequency='1',
        period='10',
        periodType='year')  # Change this for more equity information

    def ticker_selection(**kwargs):
        url = 'https://api.tdameritrade.com/v1/marketdata/{}/pricehistory' \
            .format(kwargs.get('symbol'))

        params = dict(apikey=os.environ.get('TD_API'))

        for arg in kwargs:
            params.update({arg: kwargs.get(arg)})
 
        request_values = requests.get(url=url, params=params).json()
        dataframe = pd.DataFrame(request_values.get('candles'))

        dataframe.datetime = pd.to_datetime(pd.to_datetime(dataframe.datetime, yearfirst=True,
                                            unit='ms').dt.strftime(f'%Y-%m-%d'))
        dataframe = dataframe.set_index('datetime', drop=True)

        if difference and isinstance(difference, int):
            diff = ".diff()" * difference
            df = eval('np.log(dataframe.close)'+diff+'.dropna()')
            collection.update({kwargs.get('symbol'): df})
        else:
            collection.update({kwargs.get('symbol'): dataframe.close})

    for i in equity_list:
        ticker_selection(symbol=i, **configuration)

    return pd.DataFrame(collection)


def tsplot3(ts, title, lags=None, figsize=(12, 8)):
    """
    Visualize and inspect time series Data.
    :param ts:
    :param title:
    :param lags:
    :param figsize:
    :return:
    """
    fig = plt.figure(figsize=figsize, num=f'Data Visualization of {title}')  # Create the figure
    layout = (2, 2)
    n = len(ts.values)

    # Moments
    mean = ts @ np.ones(n).T * n**-1
    variance = (ts-mean)**2 @ np.ones(n).T * (n-1)**-1
    sd = np.sqrt(variance)
    skewness = (ts-mean)**3 @ np.ones(n).T * (n-1)**-1 * (sd**3)**-1
    kurtosis = (ts-mean)**4 @ np.ones(n).T * (n-1)**-1 * (sd**4)**-1

    # Create the subplots
    ts_axis = plt.subplot2grid(layout, (0, 0))
    hist_axis = plt.subplot2grid(layout, (0, 1))
    acf_axis = plt.subplot2grid(layout, (1, 0))
    pacf_axis = plt.subplot2grid(layout, (1, 1))

    # Aggregate the Data
    ts.plot(ax=ts_axis)
    _, bins, _ = hist_axis.hist(ts.values, density=True, bins=50, ec='skyblue')
    plot_acf(ts, lags=lags, ax=acf_axis)
    plot_pacf(ts, lags=lags, ax=pacf_axis)

    # Compute the distributions
    normal = ((((1 / (np.sqrt(2 * np.pi) * sd)) *
                np.exp(-0.5 * (1 / sd * (bins - mean))**2))))

    hist_axis.plot(bins, normal, linestyle='--', label='Normal Distribution')

    # Specs
    # Returns Plot
    ts_axis.set_title(title, fontsize=14, fontweight='bold')
    ts_axis.set(
        xlabel=f'Date', ylabel=f'Log Returns',
    )
    [ts_axis.axvline(f'{year}-01-01',
                     color='k', linestyle='--', alpha=0.2) for year in set(ts.index.year)
     if pd.to_datetime(f'{year}-01-01', format=f'%Y-%m-%d') >= ts.index[0]]

    # Histogram
    info = '\n'.join((
        f'$\mu$: {round(mean, 5)}',
        f'$\sigma$: {round(sd, 5)}',
        f'$\sigma^2$: {round(variance, 5)}',
        f'$a_3$: {round(skewness, 3)}',
        f'$a_4$: {round(kurtosis, 3)}',
    ))
    props = dict(boxstyle='round', facecolor='skyblue', alpha=0.25)

    hist_axis.text(0.05, 0.95, info, transform=hist_axis.transAxes, fontsize=10,
                   verticalalignment='top', bbox=props)
    hist_axis.set(
        title=f'Histogram', ylabel=f'Density (Count)', xlabel=f'Returns'
    )
    hist_axis.legend()

    # Autocorrelation Function (ACF) and Partial Autocorrelation Function (PACF)
    [ax.set_xlim(left=0) for ax in [acf_axis, pacf_axis]]

    sns.despine()
    fig.tight_layout()
    plt.show()

    adf = sm.tsa.stattools.adfuller(ts.values)

    print(f'ADF Test Statistic is: {round(adf[0], 2)}\n'
          f'p-value: {round(adf[1], 2)}\n'
          f'Critical Values:')
    for key, value in adf[4].items():
        print(f'\t{key}: {round(value, 2)}')

    if adf[1] < 0.05:
        print(f'\n[Reject] Rejecting the null hypothesis means that the process has no unit root, '
              f'and in turn that the time series is stationary\n'
              f'or does not have time-dependent structure.\n')
    else:
        print(f'\n[Cannot Reject] Rejecting the null hypothesis means that the process has no unit root,'
              f' and in turn that the time series is stationary\n'
              f'or does not have time -dependent structure.\n')
