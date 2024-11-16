import pandas as pd
from utils import analyze_quarterly_changes, forecast_october_volume, plot_forecast

def main():
    # Load  data
    df = pd.read_csv('take_home_task_dataset.csv')

    # Analyze quarterly changes
    h_stat, p_val = analyze_quarterly_changes(df) 
    print(f"H-statistic: {h_stat:.4f}, p-value: {p_val:.4f}")

    # Call the forecasting function
    forecast_results = forecast_october_volume(df)

    # Plot the forecast
    plot_forecast(forecast_results)

if __name__ == "__main__":
    main()
