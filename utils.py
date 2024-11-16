import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats 
from statsmodels.tsa.arima.model import ARIMA  
from statsmodels.tsa.stattools import adfuller

def analyze_quarterly_changes(df):
    """
    Analyzes if transfer volumes have changed significantly between quarters using Kruskal-Wallis H-test.
    
    The function creates quarter columns, performs Kruskal-Wallis H-test on quarterly volumes,
    and returns the test statistics.
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing 'posting_date' and 'volume_gbp' columns
        
    Returns:
    --------
    tuple
        (h_statistic, p_value)
        - h_statistic : float
            Kruskal-Wallis H-statistic. Values close to 0 indicate similar distributions
        - p_value : float 
            P-value for the test. Values < 0.05 indicate statistically significant differences
            
    Example:
    --------
    >>> h_stat, p_val = analyze_quarterly_changes(df)
    >>> print(f"H-statistic: {h_stat:.4f}, p-value: {p_val:.4f}")
    H-statistic: 0.0730, p-value: 0.9947
    """
    df['posting_date'] = pd.to_datetime(df['posting_date'])

    # Create quarter column
    df['quarter'] = df['posting_date'].dt.quarter
    df['year'] = df['posting_date'].dt.year
    df['year_quarter'] = df['year'].astype(str) + '-Q' + df['quarter'].astype(str)

    # Perform Kruskal-Wallis H-test
    quarters = []
    for quarter in df['year_quarter'].unique():
        quarters.append(df[df['year_quarter'] == quarter]['volume_gbp'])

    h_statistic, p_value = stats.kruskal(*quarters)
    
    return (h_statistic, p_value)

def forecast_october_volume(df):
    """
    Forecasts the volume for October 2023 using an ARIMA(0,1,1) model based on monthly aggregated data.
    Performs stationarity testing using the Augmented Dickey-Fuller test before forecasting.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing 'posting_date' and 'volume_gbp' columns.
        - posting_date : date column for the transactions
        - volume_gbp : numeric column containing transaction volumes in GBP

    Returns:
    --------
    dict
        A dictionary containing the following keys:
        - monthly_data : pandas.Series
            Historical monthly aggregated volume data with end-of-month timestamps
        - forecast : float
            Forecasted volume for October 2023
        - confidence_interval : tuple
            95% confidence interval as (lower_bound, upper_bound)
        - forecast_date : pd.Timestamp
            Timestamp representing October 2023 (2023-10-01)

    Notes:
    ------
    - The ARIMA model is fitted only on data up to September 2023 to avoid data leakage
    - Stationarity test results are printed to console
    """
     
    # Ensure posting_date is a datetime object and set it as the index
    df['posting_date'] = pd.to_datetime(df['posting_date'])
    df.set_index('posting_date', inplace=True)

    # Aggregate the data to monthly frequency
    monthly_data = df['volume_gbp'].resample('ME').sum()

    # Perform stationarity check using Augmented Dickey-Fuller (ADF) test
    adf_result = adfuller(monthly_data)
    is_stationary = adf_result[1] < 0.05
    print(f"ADF Test p-value: {adf_result[1]:.4f} (Stationary: {is_stationary})")
    # 0.0855 p value is slighlty above 0.05. For this use case I will assume the timeseries is stationary
    # but with more time, it would be advisable to apply differencing and compare performance.
     
    # Define and fit the ARIMA model using data up to September 2023 to avoid data leakage
    model = ARIMA(monthly_data[:'2023-09-30'], order=(0, 1, 1))
    model_fit = model.fit()

    # Forecast the volume for October 2023
    october_forecast = model_fit.get_forecast(steps=1)
    october_forecast_value = float(october_forecast.predicted_mean)
    october_conf_int = october_forecast.conf_int(alpha=0.05)  # 95% Confidence Interval

    return {
        'monthly_data': monthly_data,
        'forecast': october_forecast_value,
        'confidence_interval': (october_conf_int.iloc[0, 0], october_conf_int.iloc[0, 1]),
        'forecast_date': pd.Timestamp('2023-10-01')
    }

def plot_forecast(forecast_results):
    """
    Creates a visualization of historical transaction volumes and October 2023 forecast.
    
    Parameters:
    -----------
    forecast_results : dict
        Dictionary containing forecast information with the following keys:
        - monthly_data : pandas.Series
            Historical monthly aggregated volume data with datetime index
        - forecast : float
            Forecasted volume for October 2023
        - confidence_interval : tuple
            95% confidence interval as (lower_bound, upper_bound)
        - forecast_date : pd.Timestamp
            Timestamp for October 2023 forecast
    
    Returns:
    --------
    None
        Displays a matplotlib figure with the October 2023 forecast.
    
    """
    monthly_data = forecast_results['monthly_data']
    
    # Convert end-of-month dates to start-of-month
    monthly_data.index = monthly_data.index.to_period('M').to_timestamp()
    
    # Create a new figure
    plt.figure(figsize=(12, 6))
    
    # Convert date string to timestamp for proper slicing
    cutoff_date = pd.Timestamp('2023-09-30')
    historical_data = monthly_data[monthly_data.index <= cutoff_date]
    
    # Adjust forecast date to start of month
    forecast_date = pd.Timestamp(forecast_results['forecast_date']).to_period('M').to_timestamp()
    forecasted_data = pd.Series(
        [forecast_results['forecast']],
        index=[forecast_date]
    )
    
    # Concatenate historical and forecasted data
    combined_data = pd.concat([historical_data, forecasted_data])
    
    # Calculate date range for x-axis
    start_date = combined_data.index.min() - pd.DateOffset(days=5)
    end_date = combined_data.index.max() + pd.DateOffset(days=5)
    
    # Plot the historical data
    plt.plot(historical_data.index, historical_data.values, label='Historical Volume', marker='o', color='blue')
    
    # Highlight the forecasted value
    plt.scatter(
        forecast_date,
        forecast_results['forecast'],
        color='red', label='Forecast Value', zorder=5, s=150, edgecolor='black', linewidth=1.5
    )
    
    # Add a dotted line connecting the last historical value to the forecasted value
    plt.plot(
        [historical_data.index[-1], forecast_date],
        [historical_data.values[-1], forecast_results['forecast']],
        linestyle='--', color='gray', label='Forecast Connection'
    )
    
    # Fill the confidence interval for the forecast
    plt.fill_between(
        [forecast_date - pd.DateOffset(days=2), forecast_date + pd.DateOffset(days=2)],  # Small range around forecast_date
        forecast_results['confidence_interval'][0],
        forecast_results['confidence_interval'][1],
        color='red', alpha=0.3, label='95% Confidence Interval'
    )
    
    # Set axis limits
    plt.xlim(start_date, end_date)
    plt.ylim(bottom=0) 
    
    # Format x-axis with monthly ticks
    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    
    # Format y-axis to show millions with comma separator
    def millions_formatter(x, pos):
        return f'£{x/1e6:,.1f}M'
    ax.yaxis.set_major_formatter(plt.FuncFormatter(millions_formatter))
    
    plt.gcf().autofmt_xdate()
    
    # Add labels and title
    plt.title('October 2023 Monthly Volume Forecast')
    plt.xlabel('Date')
    plt.ylabel('Transactions Volume')
    plt.legend()
    plt.grid(True)
    
    plt.margins(x=0.05)
    
    plt.tight_layout()
    plt.show()