# Basic Time Series Modelling
Apply basic time series/econometric techniques to financial data (i.e Equities, Economic Indicators).

## How to use
On the **run.py** section of the code, you can select to lookup economic time series from the St. Louis Federal Reserve website (*FRED*) or equities from the TD Ameritrade API.

### The code for equities will look like this:
```python
if __name__ == "__main"":
  equities = ['AAPL']  # here you can input any other ticker symbol for equities as a list.
  df = equity_lookup(equities)
  _df = equity_lookup(equities, difference=1)  # Use difference for stationarity.
  
  [tsplot3(_df.loc[:, series], series, lags=24) for series in _df]  # Plot the time series.
  
  arima, garch, rescale = (4, 2, 3), (2, 0, 1), 100  # Specifications

  model(df, 'Equities', log_values=True, rescale=rescale, auto_solve=True)  
  # Only one equity time series is available.
  # If auto_solve=False, you will have to manually set the arima and garch specifications to the model.
```

### For economic time series from *FRED*



