# Basic Time Series Modelling
Apply basic time series/econometric techniques to financial data (i.e Equities, Economic Indicators).

## How to use
On the **run.py** section of the code, you can select to lookup economic time series from the St. Louis Federal Reserve website (*FRED*) or equities from the TD Ameritrade API.

You may like to visit [here](https://www.investing.com/equities/most-active-stocks) for a list of the most active stocks by Investing.

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

### For economic time series from *FRED*:

Please visit the St. Louis Fed website found [here](https://fred.stlouisfed.org/).

![On top of the page you'll find the search bar for time series data.](https://github.com/sirHamilton/basic_time_series_modelling/blob/main/screenshots/fred_website.png "St. Louis Federal Reserve Page")

In the search bar you will search for an economic series you may want to inspect, let's search for *real gdp*, select the first time series that appears and save the symbol code next to the time series title.

![Real GDP search.](https://github.com/sirHamilton/basic_time_series_modelling/blob/main/screenshots/rgdp.png "RGDP Search")
![Real GDP selection and title name.](https://github.com/sirHamilton/basic_time_series_modelling/blob/main/screenshots/rgdp_name.png "Real Gross Domestic Product Time Series")

### We will use this symbol code as follows:
The following code will behave exactly as the equity one, the main difference is that it will use the *FRED* API to search for economic time series instead of the *TD Ameritrade* API for equity search. All others aspects are the same.

```python
if __name__ == "__main"":
  df = economics_series('GDPC1', end_date="2020-01-01", log=False) 
  _df = economics_series('GDPC1', end_date="2020-01-01", difference=1)

  tsplot3(_df, 'Real Gross Domestic Product', lags=24)

  arima, garch, rescale = (4, 2, 3), (2, 0, 1), 100  # Specifications

  model(df, 'Real Gross Domestic Product', log_values=True, rescale=rescale, auto_solve=True)
```

## Output

The output of our economic model, if __auto_solve__ is selected, will look like the following.

### Exploratory Data Analysis

![Exploratory Data Analysis.](https://github.com/sirHamilton/basic_time_series_modelling/blob/main/screenshots/data_visualization.png "Exploratory Data Analysis")

### Forecast and Residuals Analysis

![Forecast and Residuals Analysis.](https://github.com/sirHamilton/basic_time_series_modelling/blob/main/screenshots/output.png "Forecast and Residuals Analysis")

