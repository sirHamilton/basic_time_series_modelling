# from methods.tsPlot import equity_lookup
from methods.tsPlot import economics_series
from methods.tsPlot import tsplot3
from methods.tsForecast import model

import warnings
warnings.filterwarnings('ignore')

if __name__ == "__main__":
    # equities = ['AAPL']
    # df = equity_lookup(equities)
    # _df = equity_lookup(equities, difference=1)

    # [tsplot3(_df.loc[:, series], series, lags=24) for series in _df]  # Equities

    df = economics_series('GFDEGDQ188S', end_date="2020-01-01", log=False)
    _df = economics_series('GFDEGDQ188S', end_date="2020-01-01", difference=2)

    tsplot3(_df, 'Federal Debt: Total Public Debt as % of GDP', lags=24)

    arima, garch, rescale = (4, 2, 3), (2, 0, 1), 100  # Specifications

    model(df, 'Federal Debt: Total Public Debt as % of GDP', log_values=True, rescale=rescale, auto_solve=True)
